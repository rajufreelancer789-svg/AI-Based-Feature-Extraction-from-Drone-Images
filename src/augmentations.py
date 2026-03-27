"""
Augmentation Pipeline — Heavy augmentation for satellite/drone imagery.
Uses albumentations for geo-specific transforms.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Tuple

from src.config import TrainConfig, NUM_SEG_CLASSES


# ============================================================
# AUGMENTATION PIPELINES
# ============================================================

def get_train_transform(config: TrainConfig = TrainConfig()) -> A.Compose:
    """
    Heavy training augmentation pipeline.
    Designed for drone orthophoto segmentation.
    """
    p = config.augment_prob
    
    return A.Compose([
        # ---- Spatial Transforms ----
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Rotation + scale (simulates drone angle variation)
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.8, 1.2),
            rotate=(-45, 45),
            mode=0,  # BORDER_CONSTANT
            p=p
        ),
        
        # Elastic / Grid distortion (structural variation)
        A.OneOf([
            A.ElasticTransform(alpha=30, sigma=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.OpticalDistortion(distort_limit=0.1, p=0.3),
        ], p=0.3),
        
        # Random crop + resize (multi-scale training)
        A.OneOf([
            A.RandomResizedCrop(
                size=(config.tile_size, config.tile_size),
                scale=(0.56, 1.0),
                ratio=(0.75, 1.33),
                p=0.4
            ),
            A.NoOp(p=0.6),
        ], p=0.4),
        
        # ---- Color / Radiometric Transforms ----
        # Drone imagery has lighting variation, shadows, color shifts
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        ], p=p),
        
        # Color jittering (simulate different lighting conditions)
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05,
            p=0.3
        ),
        
        # Simulate shadows & overexposure
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 3),
            p=0.2
        ),
        
        # ---- Noise & Blur (sensor noise, atmospheric haze) ----
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=0.2),
        ], p=0.3),
        
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.MotionBlur(blur_limit=5, p=0.1),
            A.MedianBlur(blur_limit=3, p=0.1),
        ], p=0.2),
        
        # ---- Dropout (simulate occlusion, missing data) ----
        A.OneOf([
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0,
                p=0.2
            ),
            A.PixelDropout(dropout_prob=0.01, p=0.1),
        ], p=0.15),
        
        # ---- Normalize & Convert ----
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], additional_targets={'roof_mask': 'mask'})


def get_val_transform(config: TrainConfig = TrainConfig()) -> A.Compose:
    """Validation transform — normalize only, no augmentation."""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], additional_targets={'roof_mask': 'mask'})


def get_tta_transforms() -> list:
    """Test-time augmentation transforms for inference."""
    return [
        A.Compose([A.NoOp()]),                     # Original
        A.Compose([A.HorizontalFlip(p=1.0)]),      # H-Flip
        A.Compose([A.VerticalFlip(p=1.0)]),         # V-Flip
        A.Compose([A.Transpose(p=1.0)]),            # Transpose (=rot90)
        A.Compose([                                 # H-Flip + V-Flip = Rot180
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
        ]),
    ]


# ============================================================
# MOSAIC AUGMENTATION (YOLOv5-style for segmentation)
# ============================================================

def make_mosaic(
    images: list, masks: list, roof_masks: list,
    tile_size: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2x2 mosaic from 4 image-mask pairs.
    Significant data augmentation for rare-class boosting.
    
    Args:
        images: List of 4 images (H, W, 3)
        masks: List of 4 segmentation masks (H, W)
        roof_masks: List of 4 roof type masks (H, W)
        tile_size: Output mosaic size
    
    Returns:
        (mosaic_image, mosaic_seg_mask, mosaic_roof_mask)
    """
    assert len(images) == 4
    half = tile_size // 2
    
    mosaic_img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    mosaic_seg = np.zeros((tile_size, tile_size), dtype=np.uint8)
    mosaic_roof = np.zeros((tile_size, tile_size), dtype=np.uint8)
    
    # Random center point for mosaic
    cx = np.random.randint(half - half // 4, half + half // 4)
    cy = np.random.randint(half - half // 4, half + half // 4)
    
    positions = [
        (0, 0, cx, cy),                           # Top-left
        (cx, 0, tile_size, cy),                    # Top-right
        (0, cy, cx, tile_size),                    # Bottom-left
        (cx, cy, tile_size, tile_size),            # Bottom-right
    ]
    
    for i, (x1, y1, x2, y2) in enumerate(positions):
        h, w = y2 - y1, x2 - x1
        img = images[i]
        
        # Crop from the image
        src_h, src_w = img.shape[:2]
        crop_h = min(h, src_h)
        crop_w = min(w, src_w)
        
        mosaic_img[y1:y1+crop_h, x1:x1+crop_w] = img[:crop_h, :crop_w]
        mosaic_seg[y1:y1+crop_h, x1:x1+crop_w] = masks[i][:crop_h, :crop_w]
        mosaic_roof[y1:y1+crop_h, x1:x1+crop_w] = roof_masks[i][:crop_h, :crop_w]
    
    return mosaic_img, mosaic_seg, mosaic_roof


# ============================================================
# COPY-PASTE AUGMENTATION (for rare classes: utility, bridge)
# ============================================================

def copy_paste_rare_class(
    image: np.ndarray, mask: np.ndarray, roof_mask: np.ndarray,
    donor_image: np.ndarray, donor_mask: np.ndarray, donor_roof_mask: np.ndarray,
    target_classes: tuple = (4, 5),  # utility, bridge
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Copy-paste instances of rare classes from donor to target image.
    Crucial for handling extreme class imbalance (utility: 467 vs buildings: 4790).
    
    Args:
        image, mask, roof_mask: Target image and masks
        donor_image, donor_mask, donor_roof_mask: Source image with rare class instances
        target_classes: Class IDs to copy (default: utility + bridge)
    
    Returns:
        (augmented_image, augmented_mask, augmented_roof_mask)
    """
    image = image.copy()
    mask = mask.copy()
    roof_mask = roof_mask.copy()
    
    for cls_id in target_classes:
        # Find connected components of the target class in donor
        cls_pixels = donor_mask == cls_id
        if not cls_pixels.any():
            continue
        
        # Get bounding box of the class region
        rows = np.any(cls_pixels, axis=1)
        cols = np.any(cls_pixels, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Extract patch
        patch_img = donor_image[rmin:rmax+1, cmin:cmax+1].copy()
        patch_mask = donor_mask[rmin:rmax+1, cmin:cmax+1].copy()
        patch_roof = donor_roof_mask[rmin:rmax+1, cmin:cmax+1].copy()
        patch_cls = cls_pixels[rmin:rmax+1, cmin:cmax+1]
        
        # Random paste location
        ph, pw = patch_img.shape[:2]
        if ph >= image.shape[0] or pw >= image.shape[1]:
            continue
        
        py = np.random.randint(0, image.shape[0] - ph)
        px = np.random.randint(0, image.shape[1] - pw)
        
        # Paste only class pixels (alpha blending at boundary)
        image[py:py+ph, px:px+pw][patch_cls] = patch_img[patch_cls]
        mask[py:py+ph, px:px+pw][patch_cls] = patch_mask[patch_cls]
        roof_mask[py:py+ph, px:px+pw][patch_cls] = patch_roof[patch_cls]
    
    return image, mask, roof_mask
