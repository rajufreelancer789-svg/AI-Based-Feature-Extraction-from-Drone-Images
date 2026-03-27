"""
Loss Functions — Dice + Focal + Boundary for semantic segmentation.
Handles extreme class imbalance in drone imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.config import NUM_SEG_CLASSES


# ============================================================
# DICE LOSS — Region-based, handles class imbalance
# ============================================================

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    Works well when background dominates (>90% pixels).
    """
    
    def __init__(self, num_classes: int = NUM_SEG_CLASSES, smooth: float = 1.0,
                 class_weights: Optional[list] = None, ignore_index: int = -1):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw predictions
            targets: (B, H, W) class indices
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        # One-hot encode targets
        targets_oh = F.one_hot(targets.long(), self.num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Compute per-class dice
        dims = (0, 2, 3)  # Reduce over batch, height, width
        intersection = (probs * targets_oh).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Weight classes
        if self.class_weights is not None:
            weights = self.class_weights.to(dice_per_class.device)
            dice_loss = 1.0 - (dice_per_class * weights).sum() / weights.sum()
        else:
            dice_loss = 1.0 - dice_per_class.mean()
        
        return dice_loss


# ============================================================
# FOCAL LOSS — Hard example mining
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Down-weights easy examples, focuses on hard ones.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.25,
                 class_weights: Optional[list] = None, ignore_index: int = -1,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        ce_loss = F.cross_entropy(
            logits, targets.long(),
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )  # (B, H, W)
        
        # Compute focal modulation
        log_pt = -ce_loss
        pt = torch.exp(log_pt)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight
        
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


# ============================================================
# BOUNDARY LOSS — Penalize boundary errors specifically
# ============================================================

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that penalizes errors near class boundaries.
    Critical for clean building footprints and road edges.
    """
    
    def __init__(self, num_classes: int = NUM_SEG_CLASSES, boundary_width: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.boundary_width = boundary_width
        
        # Laplacian kernel for edge detection
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('edge_kernel', kernel)
    
    def _extract_boundaries(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary pixels from a segmentation mask."""
        # One-hot encode
        oh = F.one_hot(mask.long(), self.num_classes).permute(0, 3, 1, 2).float()
        
        # Apply edge detection per class
        boundaries = torch.zeros_like(mask, dtype=torch.float32)
        for c in range(self.num_classes):
            class_map = oh[:, c:c+1]  # (B, 1, H, W)
            edges = F.conv2d(class_map, self.edge_kernel, padding=1)
            boundaries += edges.squeeze(1).abs()
        
        # Dilate boundaries
        if self.boundary_width > 1:
            k = self.boundary_width * 2 + 1
            boundaries = F.max_pool2d(
                boundaries.unsqueeze(1),
                kernel_size=k, stride=1, padding=k // 2
            ).squeeze(1)
        
        return (boundaries > 0).float()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary-weighted cross-entropy loss.
        """
        # Get boundary pixels
        boundary_mask = self._extract_boundaries(targets)
        
        # Standard CE loss
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        
        # Weight boundary pixels higher (3x)
        weights = 1.0 + 2.0 * boundary_mask
        weighted_loss = (ce_loss * weights).mean()
        
        return weighted_loss


# ============================================================
# COMBINED LOSS — Dice + Focal + Boundary
# ============================================================

class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal Loss.
    Best performing loss for imbalanced segmentation.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_SEG_CLASSES,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        class_weights: Optional[list] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice = DiceLoss(num_classes=num_classes, class_weights=class_weights)
        self.focal = FocalLoss(
            gamma=focal_gamma, alpha=focal_alpha,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        d = self.dice(logits, targets)
        f = self.focal(logits, targets)
        return self.dice_weight * d + self.focal_weight * f


class DiceFocalBoundaryLoss(nn.Module):
    """
    Dice + Focal + Boundary Loss — maximum accuracy configuration.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_SEG_CLASSES,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        boundary_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        class_weights: Optional[list] = None,
        label_smoothing: float = 0.05,
        boundary_width: int = 3,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.dice = DiceLoss(num_classes=num_classes, class_weights=class_weights)
        self.focal = FocalLoss(
            gamma=focal_gamma, alpha=focal_alpha,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        self.boundary = BoundaryLoss(num_classes=num_classes, boundary_width=boundary_width)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        d = self.dice(logits, targets)
        f = self.focal(logits, targets)
        b = self.boundary(logits, targets)
        return self.dice_weight * d + self.focal_weight * f + self.boundary_weight * b


# ============================================================
# LOSS FACTORY
# ============================================================

def build_loss(config=None) -> nn.Module:
    """Build loss function from config."""
    if config is None:
        from src.config import TrainConfig
        config = TrainConfig()
    
    loss_name = config.loss_fn.lower()
    
    if loss_name == 'dice_focal':
        return DiceFocalLoss(
            num_classes=config.num_classes,
            dice_weight=config.dice_weight,
            focal_weight=config.focal_weight,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            class_weights=config.class_weights,
            label_smoothing=config.label_smoothing,
        )
    elif loss_name == 'boundary_dice_focal':
        return DiceFocalBoundaryLoss(
            num_classes=config.num_classes,
            dice_weight=config.dice_weight,
            focal_weight=config.focal_weight,
            boundary_weight=config.boundary_weight,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
            class_weights=config.class_weights,
            label_smoothing=config.label_smoothing,
        )
    elif loss_name == 'dice_ce':
        return DiceFocalLoss(
            num_classes=config.num_classes,
            focal_gamma=0.0,  # gamma=0 → standard CE
            focal_alpha=None,
            class_weights=config.class_weights,
        )
    else:
        raise ValueError(f'Unknown loss: {loss_name}')
