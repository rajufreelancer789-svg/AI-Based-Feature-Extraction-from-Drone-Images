"""
Evaluation — Comprehensive metrics for competition scoring.
Per-class IoU, F1, boundary IoU, confusion matrix, visual reports.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.config import (
    EvalConfig, SEG_CLASSES, NUM_SEG_CLASSES, ROOF_TYPES,
    OUTPUT_DIR, DRIVE_OUTPUTS, IS_COLAB,
)


# ============================================================
# PIXEL-LEVEL METRICS
# ============================================================

class SegmentationEvaluator:
    """
    Comprehensive segmentation evaluation.
    Tracks confusion matrix and computes all required metrics.
    """
    
    def __init__(self, num_classes: int = NUM_SEG_CLASSES, config: EvalConfig = EvalConfig()):
        self.num_classes = num_classes
        self.config = config
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.boundary_tp = np.zeros(self.num_classes, dtype=np.int64)
        self.boundary_fp = np.zeros(self.num_classes, dtype=np.int64)
        self.boundary_fn = np.zeros(self.num_classes, dtype=np.int64)
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Update metrics with a prediction-target pair.
        
        Args:
            pred: (H, W) predicted class indices
            target: (H, W) ground truth class indices
        """
        assert pred.shape == target.shape, f'Shape mismatch: {pred.shape} vs {target.shape}'
        
        # Flatten
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Update confusion matrix
        valid = target_flat < self.num_classes
        pred_valid = pred_flat[valid]
        target_valid = target_flat[valid]
        
        np.add.at(self.confusion_matrix, (target_valid, pred_valid), 1)
        
        # Update boundary metrics
        if 'boundary_iou' in self.config.metrics:
            self._update_boundary_metrics(pred, target)
    
    def _update_boundary_metrics(self, pred: np.ndarray, target: np.ndarray):
        """Compute boundary-specific metrics."""
        from scipy import ndimage
        
        width = self.config.boundary_width
        
        for cls_id in range(self.num_classes):
            # Extract boundaries from target and prediction
            target_boundary = self._extract_boundary(target == cls_id, width)
            pred_boundary = self._extract_boundary(pred == cls_id, width)
            
            self.boundary_tp[cls_id] += np.logical_and(target_boundary, pred_boundary).sum()
            self.boundary_fp[cls_id] += np.logical_and(~target_boundary, pred_boundary).sum()
            self.boundary_fn[cls_id] += np.logical_and(target_boundary, ~pred_boundary).sum()
    
    @staticmethod
    def _extract_boundary(mask: np.ndarray, width: int = 3) -> np.ndarray:
        """Extract boundary pixels from a binary mask."""
        from scipy import ndimage
        
        eroded = ndimage.binary_erosion(mask, iterations=width)
        boundary = mask & ~eroded
        return boundary
    
    # ---- COMPUTE METRICS ----
    
    def compute(self) -> Dict:
        """Compute all evaluation metrics."""
        cm = self.confusion_matrix
        results = {}
        
        # 1. Pixel Accuracy
        results['pixel_accuracy'] = float(np.diag(cm).sum() / (cm.sum() + 1e-8))
        
        # 2. Per-class IoU
        iou_per_class = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-8)
        
        # 3. Per-class Precision, Recall, F1
        precision_per_class = np.diag(cm) / (cm.sum(axis=0) + 1e-8)
        recall_per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
        f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-8)
        
        # 4. Foreground (classes 1-5) means
        fg = list(range(1, self.num_classes))
        results['mean_iou'] = float(iou_per_class[fg].mean())
        results['mean_f1'] = float(f1_per_class[fg].mean())
        results['mean_precision'] = float(precision_per_class[fg].mean())
        results['mean_recall'] = float(recall_per_class[fg].mean())
        
        # 5. Per-class details
        results['per_class'] = {}
        for cls_id, cls_name in SEG_CLASSES.items():
            results['per_class'][cls_name] = {
                'iou': float(iou_per_class[cls_id]),
                'precision': float(precision_per_class[cls_id]),
                'recall': float(recall_per_class[cls_id]),
                'f1': float(f1_per_class[cls_id]),
                'support': int(cm[cls_id].sum()),
            }
        
        # 6. Boundary IoU
        if 'boundary_iou' in self.config.metrics:
            boundary_iou = self.boundary_tp / (
                self.boundary_tp + self.boundary_fp + self.boundary_fn + 1e-8
            )
            results['boundary_iou'] = float(boundary_iou[fg].mean())
            for cls_id, cls_name in SEG_CLASSES.items():
                results['per_class'][cls_name]['boundary_iou'] = float(boundary_iou[cls_id])
        
        # 7. Confusion matrix
        results['confusion_matrix'] = cm.tolist()
        
        # 8. Target achievement
        results['target_achieved'] = results['mean_iou'] >= self.config.target_accuracy
        results['target_accuracy'] = self.config.target_accuracy
        
        return results
    
    def print_report(self, results: Optional[Dict] = None):
        """Pretty-print evaluation report."""
        if results is None:
            results = self.compute()
        
        print(f'\n{"="*70}')
        print(f'EVALUATION REPORT')
        print(f'{"="*70}')
        print(f'  Pixel Accuracy:  {results["pixel_accuracy"]:.4f}')
        print(f'  Mean IoU:        {results["mean_iou"]:.4f}  {"✓" if results["target_achieved"] else "✗"} (target: {results["target_accuracy"]:.2f})')
        print(f'  Mean F1:         {results["mean_f1"]:.4f}')
        print(f'  Mean Precision:  {results["mean_precision"]:.4f}')
        print(f'  Mean Recall:     {results["mean_recall"]:.4f}')
        
        if 'boundary_iou' in results:
            print(f'  Boundary IoU:    {results["boundary_iou"]:.4f}')
        
        print(f'\n  {"Class":<15} {"IoU":>8} {"F1":>8} {"Prec":>8} {"Recall":>8} {"Support":>10}')
        print(f'  {"-"*55}')
        for cls_name, cls_metrics in results['per_class'].items():
            print(
                f'  {cls_name:<15} '
                f'{cls_metrics["iou"]:>8.4f} '
                f'{cls_metrics["f1"]:>8.4f} '
                f'{cls_metrics["precision"]:>8.4f} '
                f'{cls_metrics["recall"]:>8.4f} '
                f'{cls_metrics["support"]:>10d}'
            )
        
        print(f'{"="*70}\n')
    
    def save_report(self, output_dir: str, name: str = 'evaluation'):
        """Save evaluation report to JSON."""
        results = self.compute()
        
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{name}_report.json')
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'Report saved: {path}')
        
        # Also save to Drive
        if IS_COLAB and DRIVE_OUTPUTS:
            os.makedirs(DRIVE_OUTPUTS, exist_ok=True)
            drive_path = os.path.join(DRIVE_OUTPUTS, f'{name}_report.json')
            with open(drive_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results


# ============================================================
# VILLAGE-LEVEL EVALUATION
# ============================================================

def evaluate_village(
    pred_path: str,
    gt_shp_dir: str,
    ortho_path: str,
    config: EvalConfig = EvalConfig(),
) -> Dict:
    """
    Evaluate predictions for a single village against ground truth shapefiles.
    
    Args:
        pred_path: Path to prediction raster (COG)
        gt_shp_dir: Directory with ground truth shapefiles
        ortho_path: Path to the original orthophoto
    
    Returns:
        Evaluation results dict
    """
    import rasterio
    from src.data_pipeline import load_shapefiles, create_segmentation_mask
    from src.config import TileConfig
    
    evaluator = SegmentationEvaluator(config=config)
    tile_config = TileConfig()
    
    # Load prediction
    with rasterio.open(pred_path) as src:
        pred = src.read(1)  # (H, W)
        pred_crs = str(src.crs)
        pred_transform = src.transform
    
    # Load ground truth shapefiles and rasterize
    with rasterio.open(ortho_path) as src:
        target_crs = str(src.crs)
        gt_transform = src.transform
        gt_shape = (src.height, src.width)
        bounds = src.bounds
    
    shapefiles = load_shapefiles(gt_shp_dir, target_crs)
    
    gt_mask = create_segmentation_mask(
        shapefiles,
        (bounds.left, bounds.bottom, bounds.right, bounds.top),
        gt_transform, gt_shape, tile_config
    )
    
    # Ensure same size
    h = min(pred.shape[0], gt_mask.shape[0])
    w = min(pred.shape[1], gt_mask.shape[1])
    
    # Evaluate in chunks (full image may be too large for RAM)
    chunk_size = 4096
    for r in range(0, h, chunk_size):
        for c in range(0, w, chunk_size):
            rr = min(r + chunk_size, h)
            cc = min(c + chunk_size, w)
            evaluator.update(pred[r:rr, c:cc], gt_mask[r:rr, c:cc])
    
    results = evaluator.compute()
    evaluator.print_report(results)
    
    return results


# ============================================================
# CROSS-VALIDATION
# ============================================================

def leave_one_village_out_cv(
    village_data: Dict[str, Dict],
    output_dir: str = OUTPUT_DIR,
) -> Dict:
    """
    Leave-one-village-out cross-validation.
    Trains on N-1 villages, tests on the held-out village.
    Reports per-village and aggregate metrics.
    
    Args:
        village_data: Dict mapping village_name → {
            'ortho': path_to_orthophoto,
            'shp_dir': path_to_shapefiles,
        }
        output_dir: Output directory
    
    Returns:
        Cross-validation results
    """
    village_names = sorted(village_data.keys())
    all_results = {}
    
    for val_village in village_names:
        print(f'\n{"="*50}')
        print(f'CV Fold: Validation on {val_village}')
        print(f'{"="*50}')
        
        # In a full implementation, this would retrain the model
        # For now, this is a framework for evaluation
        # The actual training loop would be called here
        
        # Placeholder for evaluation results
        all_results[val_village] = {
            'message': f'Evaluate model trained without {val_village} on {val_village}'
        }
    
    # Aggregate
    print(f'\nCross-validation complete for {len(village_names)} villages')
    
    return all_results


# ============================================================
# PLOT UTILITIES (for Colab notebooks)
# ============================================================

def plot_confusion_matrix(results: Dict, save_path: Optional[str] = None):
    """Plot confusion matrix as a heatmap."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print('matplotlib/seaborn not available for plotting')
        return
    
    cm = np.array(results['confusion_matrix'])
    class_names = list(SEG_CLASSES.values())
    
    # Normalize
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Confusion matrix saved: {save_path}')
    
    plt.show()


def plot_training_history(history: List[Dict], save_path: Optional[str] = None):
    """Plot training curves (loss, mIoU, F1)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available for plotting')
        return
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train']['loss'] for h in history]
    val_loss = [h['val']['loss'] for h in history]
    train_miou = [h['train']['mean_iou'] for h in history]
    val_miou = [h['val']['mean_iou'] for h in history]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(epochs, train_loss, label='Train')
    axes[0].plot(epochs, val_loss, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # mIoU
    axes[1].plot(epochs, train_miou, label='Train')
    axes[1].plot(epochs, val_miou, label='Val')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='Target (0.95)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Mean IoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history[0].get('train', {}):
        lr = [h['train']['lr'] for h in history]
        axes[2].plot(epochs, lr)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('LR')
        axes[2].set_title('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Training curves saved: {save_path}')
    
    plt.show()
