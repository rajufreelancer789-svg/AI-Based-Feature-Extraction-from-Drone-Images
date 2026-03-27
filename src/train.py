"""
Training Loop — Full training pipeline with AMP, EMA, checkpointing.
Designed to run on Google Colab Pro GPU (T4/A100).
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from src.config import (
    TrainConfig, TileConfig, NUM_SEG_CLASSES,
    TILES_DIR, TILES_IMAGES, TILES_MASKS,
    CHECKPOINTS_DIR, DRIVE_CHECKPOINTS, LOGS_DIR,
    IS_COLAB, DRIVE_TILES,
)
from src.model import build_model, EMAModel
from src.losses import build_loss
from src.augmentations import get_train_transform, get_val_transform, make_mosaic
from src.data_pipeline import get_train_val_split


# ============================================================
# TILE DATASET
# ============================================================

class TileDataset(Dataset):
    """
    PyTorch Dataset for pre-tiled images and masks (.npy files).
    """
    
    def __init__(
        self,
        tile_ids: list,
        images_dir: str,
        masks_dir: str,
        transform=None,
        use_mosaic: bool = False,
        mosaic_prob: float = 0.3,
        tile_size: int = 512,
    ):
        self.tile_ids = tile_ids
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob
        self.tile_size = tile_size
    
    def __len__(self):
        return len(self.tile_ids)
    
    def _load_tile(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single tile's image and masks."""
        tile_id = self.tile_ids[idx]
        
        img = np.load(os.path.join(self.images_dir, f'{tile_id}.npy'))   # (3, H, W) or (C, H, W)
        masks = np.load(os.path.join(self.masks_dir, f'{tile_id}.npy'))  # (2, H, W)
        
        # Convert image from (C, H, W) to (H, W, C) for albumentations
        if img.ndim == 3 and img.shape[0] in (3, 4):
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        
        # Take RGB only
        if img.shape[-1] == 4:
            img = img[..., :3]
        
        seg_mask = masks[0]   # (H, W)
        roof_mask = masks[1]  # (H, W)
        
        return img, seg_mask, roof_mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, seg_mask, roof_mask = self._load_tile(idx)
        
        # Mosaic augmentation (combine 4 tiles)
        if self.use_mosaic and np.random.random() < self.mosaic_prob:
            images, seg_masks, roof_masks = [img], [seg_mask], [roof_mask]
            for _ in range(3):
                rand_idx = np.random.randint(len(self))
                i, s, r = self._load_tile(rand_idx)
                images.append(i)
                seg_masks.append(s)
                roof_masks.append(r)
            
            img, seg_mask, roof_mask = make_mosaic(
                images, seg_masks, roof_masks, self.tile_size
            )
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=img,
                mask=seg_mask,
                roof_mask=roof_mask,
            )
            img = transformed['image']           # (3, H, W) tensor
            seg_mask = transformed['mask']        # (H, W) tensor
            roof_mask = transformed['roof_mask']  # (H, W) tensor
        
        return {
            'image': img,
            'seg_mask': seg_mask.long(),
            'roof_mask': roof_mask.long(),
        }


# ============================================================
# METRICS
# ============================================================

class SegMetrics:
    """
    Track segmentation metrics across batches.
    Computes: pixel_acc, mean_iou, per_class_iou, f1.
    """
    
    def __init__(self, num_classes: int = NUM_SEG_CLASSES):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update confusion matrix with batch predictions."""
        preds = preds.argmax(dim=1).cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # Filter valid targets
        valid = targets < self.num_classes
        preds = preds[valid]
        targets = targets[valid]
        
        # Update confusion matrix (vectorized)
        np.add.at(self.confusion_matrix, (targets, preds), 1)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics from confusion matrix."""
        cm = self.confusion_matrix
        
        # Pixel accuracy
        pixel_acc = np.diag(cm).sum() / (cm.sum() + 1e-8)
        
        # Per-class IoU
        iou_per_class = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-8)
        
        # Per-class F1
        precision = np.diag(cm) / (cm.sum(axis=0) + 1e-8)
        recall = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
        f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Mean (excluding background)
        fg_classes = list(range(1, self.num_classes))
        mean_iou = iou_per_class[fg_classes].mean() if len(fg_classes) > 0 else 0.0
        mean_f1 = f1_per_class[fg_classes].mean() if len(fg_classes) > 0 else 0.0
        
        from src.config import SEG_CLASSES
        
        result = {
            'pixel_acc': float(pixel_acc),
            'mean_iou': float(mean_iou),
            'mean_f1': float(mean_f1),
        }
        
        for i, name in SEG_CLASSES.items():
            result[f'iou_{name}'] = float(iou_per_class[i])
            result[f'f1_{name}'] = float(f1_per_class[i])
        
        return result


# ============================================================
# TRAINING ENGINE
# ============================================================

class Trainer:
    """
    Full training pipeline with logging, checkpointing, and early stopping.
    """
    
    def __init__(
        self,
        config: TrainConfig = TrainConfig(),
        tiles_dir: str = None,
        resume_from: str = None,
    ):
        self.config = config
        self.tiles_dir = tiles_dir or TILES_DIR
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')
        if self.device.type == 'cuda':
            print(f'  GPU: {torch.cuda.get_device_name(0)}')
            print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        
        # Build model
        self.model = build_model(config).to(self.device)
        
        # Loss
        self.criterion = build_loss(config).to(self.device)
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Scheduler
        self.scheduler = self._build_scheduler()
        
        # AMP
        self.scaler = GradScaler(enabled=config.use_amp)
        
        # EMA
        self.ema = EMAModel(self.model, config.ema_decay) if config.use_ema else None
        
        # Metrics
        self.metrics = SegMetrics()
        
        # State
        self.epoch = 0
        self.best_miou = 0.0
        self.patience_counter = 0
        self.history = []
        
        # Resume
        if resume_from:
            self._load_checkpoint(resume_from)
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer with layer-wise learning rate decay."""
        if self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        if self.config.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.epochs,
                T_mult=1,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == 'poly':
            return torch.optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=self.config.epochs,
                power=0.9,
            )
        else:
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
    
    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Build train and validation dataloaders."""
        meta_path = os.path.join(self.tiles_dir, 'dataset_meta.json')
        
        train_ids, val_ids = get_train_val_split(
            meta_path,
            train_ratio=self.config.train_split,
            seed=self.config.seed,
            strategy='village_split',
        )
        
        images_dir = os.path.join(self.tiles_dir, 'images')
        masks_dir = os.path.join(self.tiles_dir, 'masks')
        
        train_ds = TileDataset(
            train_ids, images_dir, masks_dir,
            transform=get_train_transform(self.config),
            use_mosaic=self.config.use_mosaic,
            mosaic_prob=self.config.mosaic_prob,
            tile_size=self.config.tile_size,
        )
        
        val_ds = TileDataset(
            val_ids, images_dir, masks_dir,
            transform=get_val_transform(self.config),
            use_mosaic=False,
        )
        
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.config.num_workers > 0,
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        
        print(f'Train: {len(train_ds)} tiles ({len(train_loader)} batches)')
        print(f'Val:   {len(val_ds)} tiles ({len(val_loader)} batches)')
        
        return train_loader, val_loader
    
    # ---- TRAINING LOOP ----
    
    def train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            images = batch['image'].to(self.device, non_blocking=True)
            seg_masks = batch['seg_mask'].to(self.device, non_blocking=True)
            
            # Forward pass with AMP
            with autocast(device_type=self.device.type, enabled=self.config.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs['seg_logits'], seg_masks)
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # EMA update
            if self.ema:
                self.ema.update(self.model)
            
            # Metrics
            with torch.no_grad():
                self.metrics.update(outputs['seg_logits'], seg_masks)
            
            total_loss += loss.item()
            num_batches += 1
        
        # NOTE: scheduler step moved to main train() loop to avoid warmup conflict
        
        metrics = self.metrics.compute()
        metrics['loss'] = total_loss / max(num_batches, 1)
        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        return metrics
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        torch.cuda.empty_cache()
        # Use EMA weights for validation if available
        if self.ema:
            self.ema.apply_shadow(self.model)
        
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            images = batch['image'].to(self.device, non_blocking=True)
            seg_masks = batch['seg_mask'].to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, enabled=self.config.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs['seg_logits'], seg_masks)
            
            self.metrics.update(outputs['seg_logits'], seg_masks)
            total_loss += loss.item()
            num_batches += 1
        
        metrics = self.metrics.compute()
        metrics['loss'] = total_loss / max(num_batches, 1)
        
        # Restore original weights
        if self.ema:
            self.ema.restore(self.model)
        
        return metrics
    
    # ---- MAIN TRAINING ----
    
    def train(self):
        """Full training procedure."""
        print(f'\n{"="*70}')
        print(f'TRAINING START')
        print(f'{"="*70}')
        print(f'Model: {self.config.model_name} ({self.config.backbone})')
        print(f'Epochs: {self.config.epochs}, Batch: {self.config.batch_size}')
        print(f'Loss: {self.config.loss_fn}, LR: {self.config.learning_rate}')
        print(f'AMP: {self.config.use_amp}, EMA: {self.config.use_ema}')
        print(f'{"="*70}\n')
        
        train_loader, val_loader = self._build_dataloaders()
        
        # Warmup (linear warmup for first N epochs)
        warmup_scheduler = None
        if self.config.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=self.config.warmup_epochs,
            )
        
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_one_epoch(train_loader)
            
            # LR scheduling: warmup overrides main scheduler
            if warmup_scheduler and epoch < self.config.warmup_epochs:
                warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            # Log
            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time)
            
            # Save history
            self.history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'time': epoch_time,
            })
            
            # Checkpointing
            is_best = val_metrics['mean_iou'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['mean_iou']
                self.patience_counter = 0
                self._save_checkpoint('best.pth', val_metrics)
                print(f'  ★ New best mIoU: {self.best_miou:.4f}')
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f'epoch_{epoch+1}.pth', val_metrics)
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f'\nEarly stopping at epoch {epoch+1} (patience={self.config.early_stopping_patience})')
                break
        
        # Save final
        self._save_checkpoint('final.pth', val_metrics)
        self._save_history()
        
        print(f'\n{"="*70}')
        print(f'TRAINING COMPLETE')
        print(f'Best mIoU: {self.best_miou:.4f}')
        print(f'{"="*70}')
        
        return self.history
    
    # ---- CHECKPOINTING ----
    
    def _save_checkpoint(self, name: str, val_metrics: Dict):
        """Save model checkpoint to local and Drive."""
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_miou': self.best_miou,
            'val_metrics': val_metrics,
            'config': {
                'model_name': self.config.model_name,
                'backbone': self.config.backbone,
                'num_classes': self.config.num_classes,
            },
        }
        
        if self.ema:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        path = os.path.join(CHECKPOINTS_DIR, name)
        torch.save(checkpoint, path)
        
        # Copy to Drive for persistence
        if IS_COLAB and DRIVE_CHECKPOINTS:
            os.makedirs(DRIVE_CHECKPOINTS, exist_ok=True)
            drive_path = os.path.join(DRIVE_CHECKPOINTS, name)
            torch.save(checkpoint, drive_path)
    
    def _load_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        print(f'Loading checkpoint: {path}')
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        
        if self.ema and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        print(f'  Resumed from epoch {self.epoch}, best mIoU: {self.best_miou:.4f}')
    
    def _save_history(self):
        """Save training history as JSON."""
        os.makedirs(LOGS_DIR, exist_ok=True)
        path = os.path.join(LOGS_DIR, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        if IS_COLAB and DRIVE_CHECKPOINTS:
            drive_path = os.path.join(DRIVE_CHECKPOINTS, 'training_history.json')
            with open(drive_path, 'w') as f:
                json.dump(self.history, f, indent=2)
    
    def _log_epoch(self, epoch: int, train: Dict, val: Dict, elapsed: float):
        """Pretty-print epoch results."""
        print(
            f'Epoch {epoch+1:3d}/{self.config.epochs} │ '
            f'{elapsed:5.1f}s │ '
            f'Train Loss: {train["loss"]:.4f} mIoU: {train["mean_iou"]:.4f} │ '
            f'Val Loss: {val["loss"]:.4f} mIoU: {val["mean_iou"]:.4f} F1: {val["mean_f1"]:.4f} │ '
            f'LR: {train["lr"]:.2e}'
        )
