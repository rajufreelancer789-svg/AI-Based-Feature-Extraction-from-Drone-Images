"""
Project Configuration — Single source of truth for all settings.
PS-1: AI-Based Feature Extraction from Drone Images
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ============================================================
# PATHS (adapt automatically for Colab vs Local)
# ============================================================
IS_COLAB = os.path.exists('/content')

if IS_COLAB:
    DRIVE_PROJECT = '/content/drive/MyDrive/IITT_AIML'
    DRIVE_ZIPS = '/content/drive/MyDrive/IITT_AIML/zips'
    LOCAL_PROJECT = '/content/IITT_AIML'
else:
    DRIVE_PROJECT = None
    DRIVE_ZIPS = None
    LOCAL_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.path.join(LOCAL_PROJECT, 'data')
TRAIN_DIR_CG = os.path.join(DATA_ROOT, 'train', 'CG')
TRAIN_DIR_PB = os.path.join(DATA_ROOT, 'train', 'PB')
TEST_DIR_CG = os.path.join(DATA_ROOT, 'test', 'CG')
TEST_DIR_PB = os.path.join(DATA_ROOT, 'test', 'PB')
LABELS_DIR_CG = os.path.join(DATA_ROOT, 'labels', 'CG')
LABELS_DIR_PB = os.path.join(DATA_ROOT, 'labels', 'PB')
TILES_DIR = os.path.join(DATA_ROOT, 'tiles')
TILES_IMAGES = os.path.join(TILES_DIR, 'images')
TILES_MASKS = os.path.join(TILES_DIR, 'masks')

OUTPUT_DIR = os.path.join(LOCAL_PROJECT, 'outputs')
CHECKPOINTS_DIR = os.path.join(LOCAL_PROJECT, 'checkpoints')
LOGS_DIR = os.path.join(LOCAL_PROJECT, 'logs')

# Drive-persistent directories (survive runtime restarts)
if IS_COLAB:
    DRIVE_CHECKPOINTS = os.path.join(DRIVE_PROJECT, 'checkpoints')
    DRIVE_OUTPUTS = os.path.join(DRIVE_PROJECT, 'outputs')
    DRIVE_TILES = os.path.join(DRIVE_PROJECT, 'tiles')
else:
    DRIVE_CHECKPOINTS = CHECKPOINTS_DIR
    DRIVE_OUTPUTS = OUTPUT_DIR
    DRIVE_TILES = TILES_DIR


# ============================================================
# CLASS DEFINITIONS
# ============================================================

# Semantic segmentation classes
SEG_CLASSES = {
    0: 'background',
    1: 'building',
    2: 'road',
    3: 'waterbody',
    4: 'utility',
    5: 'bridge',
    6: 'railway',
}
NUM_SEG_CLASSES = len(SEG_CLASSES)

# Rooftop sub-classification (applied only to building pixels)
ROOF_TYPES = {
    1: 'RCC',           # Reinforced Cement Concrete — grey
    2: 'Tiled',         # Clay/ceramic tiles — orange/brown
    3: 'Tin/Metal',     # Metal sheets — shiny/reflective
    4: 'Thatched',      # Straw/kutcha — brown/textured
}
NUM_ROOF_TYPES = len(ROOF_TYPES)

# Utility sub-classification
UTILITY_TYPES = {
    1: 'Distribution Transformer',
    2: 'Overhead Tank',
    3: 'Well',
    11: 'Other',
}

# Class colors for visualization (RGB)
SEG_COLORS = {
    0: (0, 0, 0),         # background — black
    1: (255, 0, 0),       # building — red
    2: (255, 255, 0),     # road — yellow
    3: (0, 0, 255),       # waterbody — blue
    4: (0, 255, 0),       # utility — green
    5: (255, 128, 0),     # bridge — orange
    6: (128, 0, 255),     # railway — purple
}

ROOF_COLORS = {
    1: (128, 128, 128),   # RCC — grey
    2: (204, 102, 0),     # Tiled — brown
    3: (192, 192, 192),   # Tin — silver
    4: (139, 90, 43),     # Thatched — dark brown
}

# Shapefile → Class mapping
# Handles both CG and PB naming conventions
SHAPEFILE_CLASS_MAP = {
    'Built_Up_Area_type': 1,  # building (CG naming)
    'Built_Up_Area_typ': 1,   # building (PB naming — truncated)
    'Road': 2,                # road
    'Road_Centre_Line': 2,    # road (centerline)
    'Water_Body': 3,          # waterbody
    'Water_Body_Line': 3,     # waterbody (line)
    'Waterbody_Point': 3,     # waterbody (point — wells etc.)
    'Utility': 4,             # utility points
    'Utility_Poly': 4,        # utility polygons (CG naming)
    'Utility_Poly_': 4,       # utility polygons (PB naming — trailing underscore)
    'Bridge': 5,              # bridge
    'Railway': 6,             # railway (empty in CG, 4 features in PB)
}

# Canonical name mapping: maps variant names → standard names
# Used to normalize PB names to CG-style names in the pipeline
SHAPEFILE_NAME_ALIASES = {
    'Built_Up_Area_typ': 'Built_Up_Area_type',
    'Utility_Poly_': 'Utility_Poly',
}


# ============================================================
# TILING CONFIGURATION
# ============================================================

@dataclass
class TileConfig:
    """Configuration for tiling orthophotos into training patches."""
    tile_size: int = 512              # Tile dimensions (pixels)
    overlap: int = 128                # Overlap between adjacent tiles
    min_labeled_ratio: float = 0.01   # Min fraction of non-background pixels to keep tile
    bands: Tuple[int, ...] = (1, 2, 3)  # RGB bands (1-indexed for rasterio)
    target_crs: str = None            # Target CRS (None = use orthophoto CRS)
    
    # Line features buffer (meters) — convert lines to polygons for rasterization
    line_buffer_m: float = 1.5        # Buffer width for road centerlines, water lines
    point_buffer_m: float = 3.0       # Buffer radius for point features (utilities, wells)
    
    @property
    def stride(self) -> int:
        return self.tile_size - self.overlap


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

@dataclass
class TrainConfig:
    """Configuration for model training."""
    # Model
    model_name: str = 'segformer'     # 'segformer', 'swin_unet', 'deeplabv3plus', 'mask2former'
    backbone: str = 'mit_b3'          # Encoder backbone
    pretrained: bool = True
    num_classes: int = NUM_SEG_CLASSES
    
    # Training
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 6e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Optimizer
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'         # 'cosine', 'poly', 'step'
    
    # Loss
    loss_fn: str = 'dice_focal'       # 'dice_focal', 'dice_ce', 'boundary_dice_focal'
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    dice_weight: float = 1.0
    focal_weight: float = 1.0
    boundary_weight: float = 0.5
    
    # Class weights (handle imbalance — background is majority)
    # Order: [bg, building, road, waterbody, utility, bridge, railway]
    class_weights: List[float] = field(default_factory=lambda: [0.1, 2.0, 1.5, 2.0, 3.0, 3.0, 3.0])
    
    # Data
    tile_size: int = 512
    train_split: float = 0.85
    val_split: float = 0.15
    num_workers: int = 4
    
    # Augmentation
    use_augmentation: bool = True
    augment_prob: float = 0.5
    use_mosaic: bool = True
    use_copypaste: bool = True        # Copy-paste rare classes (utilities, bridges)
    mosaic_prob: float = 0.3
    copypaste_prob: float = 0.3
    
    # Regularization
    drop_path_rate: float = 0.1
    label_smoothing: float = 0.05
    use_ema: bool = True              # Exponential Moving Average
    ema_decay: float = 0.9999
    
    # Checkpointing
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 15
    
    # Mixed precision
    use_amp: bool = True
    
    # Reproducibility
    seed: int = 42


# ============================================================
# INFERENCE CONFIGURATION
# ============================================================

@dataclass
class InferConfig:
    """Configuration for inference on full orthophotos."""
    tile_size: int = 512
    overlap: int = 256                # Higher overlap for inference → better stitching
    batch_size: int = 16
    use_tta: bool = True              # Test-time augmentation
    tta_transforms: List[str] = field(default_factory=lambda: ['hflip', 'vflip', 'rot90'])
    
    # Ensemble
    use_ensemble: bool = True
    ensemble_models: List[str] = field(default_factory=lambda: [
        'segformer_mit_b3',
        'swin_unet_base',
        'deeplabv3plus_resnet101'
    ])
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])
    
    # Post-processing
    confidence_threshold: float = 0.5
    min_building_area_sqm: float = 4.0    # Remove buildings < 4 sqm
    min_road_length_m: float = 5.0        # Remove road segments < 5m
    use_crf: bool = True                  # Conditional Random Field refinement
    use_morphology: bool = True           # Morphological operations (open/close)
    
    # Vectorization
    simplify_tolerance: float = 0.5       # Douglas-Peucker simplification (meters)
    
    # Output
    output_format_raster: str = 'COG'     # Cloud Optimized GeoTIFF
    output_format_vector: str = 'GPKG'    # GeoPackage
    
    # ONNX export
    export_onnx: bool = True
    onnx_opset: int = 17
    quantize_int8: bool = True


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

@dataclass
class EvalConfig:
    """Configuration for model evaluation."""
    target_accuracy: float = 0.95
    metrics: List[str] = field(default_factory=lambda: [
        'pixel_accuracy',
        'mean_iou',
        'per_class_iou',
        'f1_score',
        'precision',
        'recall',
        'boundary_iou',        # Edge quality metric
        'confusion_matrix',
    ])
    boundary_width: int = 3           # Pixels for boundary IoU calculation
    
    # Cross-validation
    use_cross_val: bool = True
    cv_strategy: str = 'leave_one_village_out'  # Best for spatial generalization


# ============================================================
# DOWNLOAD LINKS
# ============================================================

DATASETS = {
    'CG_Training_dataSet_2.zip': {
        'url': 'https://svamitva.nic.in/DownloadPDF/TifFile/CG_Training_dataSet_2.zip',
        'type': 'train', 'state': 'CG',
        'desc': 'CG Training Orthophotos Set 2 (BADETUMNAR, KUTRU)',
    },
    'CG_Training_dataSet_3.zip': {
        'url': 'https://svamitva.nic.in/DownloadPDF/TifFile/CG_Training_dataSet_3.zip',
        'type': 'train', 'state': 'CG',
        'desc': 'CG Training Orthophotos Set 3 (MURDANDA, NAGUL, SAMLUR)',
    },
    'CG_shp-file.zip': {
        'url': 'https://svamitva.nic.in/DownloadPDF/TifFile/CG_shp-file.zip',
        'type': 'labels', 'state': 'CG',
        'desc': 'CG Shapefiles (building, road, water, utility labels)',
    },
    'PB_training_dataSet_shp_file.zip': {
        'url': 'https://svamitva.nic.in/DownloadPDF/TifFile/PB_training_dataSet_shp_file.zip',
        'type': 'train', 'state': 'PB',
        'desc': 'PB Training Orthophotos + Shapefiles',
    },
    'PB_live_demo_2.zip': {
        'url': 'https://svamitva.nic.in/DownloadPDF/TifFile/PB_live_demo_2.zip',
        'type': 'test', 'state': 'PB',
        'desc': 'PB Testing Orthophotos Set 2',
    },
    'PB_live_demo_3.zip': {
        'url': 'https://svamitva.nic.in/DownloadPDF/TifFile/PB_live_demo_3.zip',
        'type': 'test', 'state': 'PB',
        'desc': 'PB Testing Orthophotos Set 3',
    },
    'CG_live-demo.zip': {
        'url': 'https://svamitva.nic.in/DownloadPDF/TifFile/CG_live-demo.zip',
        'type': 'test', 'state': 'CG',
        'desc': 'CG Testing Orthophotos',
    },
}
