"""
IITT_AIML — AI-Based Feature Extraction from Drone Images
PS-1: SVAMITVA Scheme Hackathon @ IIT Tirupati

Modules:
    config          — Project configuration (paths, classes, hyperparams)
    data_pipeline   — Tiling, mask generation, dataset creation
    augmentations   — Training augmentation pipeline (albumentations)
    model           — SegFormer, DeepLabV3+, UNet++ architectures
    losses          — Dice + Focal + Boundary combined loss
    train           — Full training loop with AMP, EMA, checkpointing
    inference       — Sliding window prediction with TTA + ensemble
    postprocess     — Morphology, vectorization, COG/GPKG export
    evaluate        — Per-class IoU, F1, boundary IoU, reports
"""

__version__ = '0.1.0'
