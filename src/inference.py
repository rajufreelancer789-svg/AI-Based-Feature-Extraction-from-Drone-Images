"""
Inference Pipeline — Full orthophoto prediction with TTA and ensemble.
Produces segmentation maps for entire villages at native resolution.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds

from src.config import (
    InferConfig, NUM_SEG_CLASSES, SEG_CLASSES,
    OUTPUT_DIR, DRIVE_OUTPUTS, IS_COLAB,
    CHECKPOINTS_DIR, DRIVE_CHECKPOINTS,
)
from src.model import build_model, EMAModel
from src.augmentations import get_val_transform


# ============================================================
# INFERENCE ENGINE
# ============================================================

class Inferencer:
    """
    Full orthophoto inference with sliding window, TTA, and ensemble.
    """
    
    def __init__(
        self,
        config: InferConfig = InferConfig(),
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.config = config
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.transform = get_val_transform()
        
        # Load model(s)
        if config.use_ensemble and checkpoint_path is None:
            self.models = self._load_ensemble()
        elif checkpoint_path:
            self.models = [self._load_single_model(checkpoint_path)]
            self.ensemble_weights = [1.0]
        else:
            raise ValueError('Provide checkpoint_path or enable ensemble with checkpoints in CHECKPOINTS_DIR')
        
        self.ensemble_weights = config.ensemble_weights[:len(self.models)]
        # Normalize weights
        w_sum = sum(self.ensemble_weights)
        self.ensemble_weights = [w / w_sum for w in self.ensemble_weights]
    
    def _load_single_model(self, path: str) -> nn.Module:
        """Load a single model from checkpoint."""
        print(f'Loading model: {path}')
        checkpoint = torch.load(path, map_location=self.device)
        
        from src.config import TrainConfig
        model_cfg = checkpoint.get('config', {})
        config = TrainConfig(
            model_name=model_cfg.get('model_name', 'segformer_simple'),
            backbone=model_cfg.get('backbone', 'mit_b3'),
            num_classes=model_cfg.get('num_classes', NUM_SEG_CLASSES),
        )
        
        model = build_model(config).to(self.device)
        
        # Load EMA weights if available, otherwise regular weights
        if 'ema_shadow' in checkpoint:
            for name, param in model.named_parameters():
                if name in checkpoint['ema_shadow']:
                    param.data = checkpoint['ema_shadow'][name].to(self.device)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        return model
    
    def _load_ensemble(self) -> List[nn.Module]:
        """Load all ensemble models from checkpoints directory."""
        models = []
        ckpt_dir = DRIVE_CHECKPOINTS if IS_COLAB else CHECKPOINTS_DIR
        
        for model_name in self.config.ensemble_models:
            path = os.path.join(ckpt_dir, f'{model_name}_best.pth')
            if os.path.exists(path):
                models.append(self._load_single_model(path))
            else:
                # Fall back to generic best.pth
                fallback = os.path.join(ckpt_dir, 'best.pth')
                if os.path.exists(fallback):
                    print(f'  Using fallback: {fallback}')
                    models.append(self._load_single_model(fallback))
                    break  # Only load once as fallback
        
        if not models:
            raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
        
        print(f'Loaded {len(models)} model(s) for ensemble')
        return models
    
    # ---- SLIDING WINDOW PREDICTION ----
    
    @torch.no_grad()
    def predict_tile(self, image: np.ndarray) -> np.ndarray:
        """
        Predict a single 512×512 tile.
        
        Args:
            image: (H, W, 3) uint8 RGB image
        
        Returns:
            (num_classes, H, W) probability map
        """
        # Apply normalization transform
        transformed = self.transform(image=image)
        x = transformed['image'].unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        # Ensemble prediction
        probs_sum = None
        
        for model, weight in zip(self.models, self.ensemble_weights):
            if self.config.use_tta:
                probs = self._predict_with_tta(model, x)
            else:
                with autocast(device_type=self.device.type, enabled=True):
                    outputs = model(x)
                probs = F.softmax(outputs['seg_logits'], dim=1)
            
            if probs_sum is None:
                probs_sum = probs * weight
            else:
                probs_sum += probs * weight
        
        return probs_sum.squeeze(0).cpu().numpy()  # (C, H, W)
    
    def _predict_with_tta(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Test-time augmentation prediction."""
        probs_list = []
        
        # Original
        with autocast(device_type=self.device.type, enabled=True):
            probs_list.append(F.softmax(model(x)['seg_logits'], dim=1))
        
        # Horizontal flip
        if 'hflip' in self.config.tta_transforms:
            x_flip = torch.flip(x, dims=[3])
            with autocast(device_type=self.device.type, enabled=True):
                p = F.softmax(model(x_flip)['seg_logits'], dim=1)
            probs_list.append(torch.flip(p, dims=[3]))
        
        # Vertical flip
        if 'vflip' in self.config.tta_transforms:
            x_flip = torch.flip(x, dims=[2])
            with autocast(device_type=self.device.type, enabled=True):
                p = F.softmax(model(x_flip)['seg_logits'], dim=1)
            probs_list.append(torch.flip(p, dims=[2]))
        
        # Rotate 90
        if 'rot90' in self.config.tta_transforms:
            x_rot = torch.rot90(x, k=1, dims=[2, 3])
            with autocast(device_type=self.device.type, enabled=True):
                p = F.softmax(model(x_rot)['seg_logits'], dim=1)
            probs_list.append(torch.rot90(p, k=-1, dims=[2, 3]))
        
        # Average all TTA predictions
        probs = torch.stack(probs_list).mean(dim=0)
        return probs
    
    # ---- FULL ORTHOPHOTO INFERENCE ----
    
    def predict_orthophoto(
        self,
        ortho_path: str,
        output_dir: str = None,
        village_name: str = None,
    ) -> Dict[str, str]:
        """
        Run inference on a full orthophoto with sliding window.
        
        Args:
            ortho_path: Path to the orthophoto GeoTIFF
            output_dir: Output directory
            village_name: Name for output files
        
        Returns:
            Dict with paths to output files
        """
        output_dir = output_dir or OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        village_name = village_name or Path(ortho_path).stem
        
        print(f'\n{"="*60}')
        print(f'INFERENCE: {village_name}')
        print(f'{"="*60}')
        
        with rasterio.open(ortho_path) as src:
            width, height = src.width, src.height
            crs = src.crs
            transform = src.transform
            print(f'  Size: {width}x{height}, CRS: {crs}')
            
            # Allocate output arrays
            prob_map = np.zeros((NUM_SEG_CLASSES, height, width), dtype=np.float32)
            count_map = np.zeros((height, width), dtype=np.float32)
            
            # Sliding window
            tile_size = self.config.tile_size
            overlap = self.config.overlap
            stride = tile_size - overlap
            
            total_tiles = ((height - 1) // stride + 1) * ((width - 1) // stride + 1)
            processed = 0
            
            for row_off in range(0, height, stride):
                for col_off in range(0, width, stride):
                    # Calculate actual window (clip to image bounds)
                    w = min(tile_size, width - col_off)
                    h = min(tile_size, height - row_off)
                    
                    window = Window(col_off, row_off, w, h)
                    tile = src.read([1, 2, 3], window=window)  # (3, h, w)
                    tile = np.transpose(tile, (1, 2, 0))  # (h, w, 3)
                    
                    # Pad to full tile size if needed
                    if h < tile_size or w < tile_size:
                        padded = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                        padded[:h, :w] = tile
                        tile = padded
                    
                    # Predict
                    pred = self.predict_tile(tile)  # (C, tile_size, tile_size)
                    
                    # Accumulate predictions (with Gaussian weighting for overlap regions)
                    weight = self._gaussian_weight(tile_size)
                    
                    # Crop to actual size
                    pred = pred[:, :h, :w]
                    weight_crop = weight[:h, :w]
                    
                    prob_map[:, row_off:row_off+h, col_off:col_off+w] += pred * weight_crop
                    count_map[row_off:row_off+h, col_off:col_off+w] += weight_crop
                    
                    processed += 1
                    if processed % 100 == 0:
                        print(f'  Progress: {processed}/{total_tiles} tiles ({100*processed/total_tiles:.0f}%)')
            
            # Normalize by count (handle divide-by-zero)
            count_map = np.maximum(count_map, 1e-8)
            prob_map /= count_map
        
        # Argmax → class map
        class_map = prob_map.argmax(axis=0).astype(np.uint8)
        confidence_map = prob_map.max(axis=0)
        
        print(f'  Prediction complete. Unique classes: {np.unique(class_map).tolist()}')
        
        # Save outputs
        output_paths = self._save_outputs(
            class_map, prob_map, confidence_map,
            ortho_path, output_dir, village_name
        )
        
        # Copy to Drive
        if IS_COLAB and DRIVE_OUTPUTS:
            os.makedirs(DRIVE_OUTPUTS, exist_ok=True)
            for key, path in output_paths.items():
                import shutil
                drive_path = os.path.join(DRIVE_OUTPUTS, os.path.basename(path))
                shutil.copy2(path, drive_path)
        
        return output_paths
    
    def _gaussian_weight(self, size: int) -> np.ndarray:
        """Create a 2D Gaussian weight map for overlap blending."""
        sigma = size / 4
        x = np.arange(size)
        y = np.arange(size)
        xx, yy = np.meshgrid(x, y)
        cx, cy = size / 2, size / 2
        weight = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        return weight.astype(np.float32)
    
    def _save_outputs(
        self,
        class_map: np.ndarray,
        prob_map: np.ndarray,
        confidence_map: np.ndarray,
        ortho_path: str,
        output_dir: str,
        village_name: str,
    ) -> Dict[str, str]:
        """Save prediction outputs as GeoTIFF (COG format)."""
        from src.postprocess import save_as_cog
        
        with rasterio.open(ortho_path) as src:
            profile = src.profile.copy()
        
        paths = {}
        
        # 1. Class map (COG)
        class_path = os.path.join(output_dir, f'{village_name}_segmentation.tif')
        save_as_cog(
            class_map, profile,
            class_path,
            nodata=255,
            dtype='uint8',
        )
        paths['segmentation'] = class_path
        
        # 2. Confidence map (COG)
        conf_path = os.path.join(output_dir, f'{village_name}_confidence.tif')
        save_as_cog(
            confidence_map, profile,
            conf_path,
            dtype='float32',
        )
        paths['confidence'] = conf_path
        
        print(f'  Saved: {class_path}')
        print(f'  Saved: {conf_path}')
        
        return paths
