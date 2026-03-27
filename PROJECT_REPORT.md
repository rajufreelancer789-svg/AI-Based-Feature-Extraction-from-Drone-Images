# PS-1: AI-Based Feature Extraction from Drone Images
## SVAMITVA Scheme — IIT Tirupati AI/ML Hackathon

---

## 1. Problem Statement

**Title:** AI-Based Feature Extraction from Drone Orthoimagery for the SVAMITVA Scheme

**Objective:** Develop a deep learning pipeline that automatically extracts geographic features from high-resolution drone orthophotos of rural Indian villages. The model must segment and identify:

| Class ID | Feature | Description |
|----------|---------|-------------|
| 0 | Background | Non-feature areas (vegetation, open land) |
| 1 | Building | Built-up structures (residential, commercial) |
| 2 | Road | Roads and road centre lines |
| 3 | Waterbody | Rivers, ponds, wells, water channels |
| 4 | Utility | Distribution transformers, overhead tanks, wells |
| 5 | Bridge | Road/rail bridges |
| 6 | Railway | Railway tracks |

**Real-World Significance:** The SVAMITVA (Survey of Villages Abadi and Mapping with Improvised Technology in Village Areas) scheme by the Government of India uses drone surveys to map rural properties. Automating feature extraction reduces manual GIS effort from weeks to hours.

---

Our QC Visualisation Output : 
<img width="1600" height="401" alt="image" src="https://github.com/user-attachments/assets/7343cd02-845c-45a0-8ed0-9c0ed7e21592" />


## 2. Dataset Description

### 2.1 Data Source
- **Provider:** Survey of India (SOI), under SVAMITVA Scheme
- **States:** Chhattisgarh (CG) and Punjab (PB)
- **Format:** High-resolution drone orthophotos (GeoTIFF) + Ground truth shapefiles

### 2.2 Training Data — 6 Villages

| # | Village Name | State | Orthophoto Resolution | Shapefiles |
|---|---|---|---|---|
| 1 | Aundhi | CG | ~5 cm/pixel | 10 shapefiles |
| 2 | Dhamansara | CG | ~5 cm/pixel | 10 shapefiles |
| 3 | Hardi-Potia-Pandripani | CG | ~5 cm/pixel | 10 shapefiles |
| 4 | Nagul-Madase-Ghotpal | CG | ~5 cm/pixel | 10 shapefiles |
| 5 | Parsada | CG | ~5 cm/pixel | 10 shapefiles |
| 6 | Sahaspur-Bangar | CG | ~5 cm/pixel | 10 shapefiles |

### 2.3 Shapefiles per Village
Each village has the following ground truth shapefiles:

| Shapefile | Mapped To | Geometry Type |
|-----------|-----------|---------------|
| Built_Up_Area_type | Building (1) | Polygon |
| Road | Road (2) | Polygon |
| Road_Centre_Line | Road (2) | Line → buffered to polygon (1.5m) |
| Water_Body | Waterbody (3) | Polygon |
| Water_Body_Line | Waterbody (3) | Line → buffered to polygon (1.5m) |
| Waterbody_Point | Waterbody (3) | Point → buffered to polygon (3.0m) |
| Utility | Utility (4) | Point → buffered to polygon (3.0m) |
| Utility_Poly | Utility (4) | Polygon |
| Bridge | Bridge (5) | Polygon |
| Railway | Railway (6) | Line (empty in CG, 4 features in PB) |

### 2.4 Class Distribution (from 8,449 tiles)

| Class | Tiles Present | % of tiles | Notes |
|-------|--------------|------------|-------|
| Building | 5,953 | 70.5% | Most common feature |
| Road | 3,964 | 46.9% | Well-represented |
| Waterbody | 2,268 | 26.8% | Moderate |
| Utility | 628 | 7.4% | Rare — needs special handling |
| Bridge | 20 | 0.2% | Very rare |
| Railway | 0 | 0.0% | No CG data (only PB) |
| Background | 8,449 | 100% | Present in all tiles (dominant class, >90% pixels) |

**Key Challenge:** Extreme class imbalance. Background dominates (>90% of pixels). Utility, bridge, and railway are extremely rare.

### 2.5 Test Data
- Separate zip files provided by organizers for inference evaluation
- Will be processed through NB04 (inference notebook)

---

## 3. Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL PIPELINE OVERVIEW                        │
│                                                                 │
│  RAW DATA                                                       │
│  ├── Orthophotos (GeoTIFF, ~10K×10K px each)                   │
│  └── Shapefiles (10 per village)                                │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                            │
│  │  NB01: DATA PREP │  Download zips → extract → build manifest │
│  └────────┬────────┘                                            │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  NB02: TILING    │  Orthophoto + Shapefiles → 512×512 tiles  │
│  │  (data_pipeline) │  8,449 image-mask pairs (.npy)            │
│  └────────┬────────┘                                            │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  NB03: TRAINING  │  SegFormer mit_b3 training                │
│  │  (train.py)      │  80 epochs, boundary_dice_focal loss      │
│  └────────┬────────┘                                            │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  NB04: INFERENCE │  Sliding window + TTA → full ortho pred   │
│  │  (inference.py)  │  Post-process → COG + GPKG output         │
│  └─────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Codebase Structure

```
IITT_AIML/
├── src/                          # Core source modules (9 files)
│   ├── __init__.py
│   ├── config.py                 # All configurations, class defs, paths
│   ├── data_pipeline.py          # Tiling, mask rasterization, train/val split
│   ├── augmentations.py          # Training augmentations (albumentations v2)
│   ├── model.py                  # SegFormer, DeepLabV3+, Swin-UNet architectures
│   ├── losses.py                 # Dice + Focal + Boundary loss functions
│   ├── train.py                  # Trainer class with full training loop
│   ├── inference.py              # Sliding window inference + TTA
│   ├── postprocess.py            # Morphological cleanup, vectorization, export
│   └── evaluate.py               # mIoU, F1, boundary IoU, confusion matrix
│
├── notebooks/                    # 4 Jupyter notebooks (run on Google Colab)
│   ├── 01_data_prep.ipynb        # Data download, extraction, manifest
│   ├── 02_tiling.ipynb           # Tile generation pipeline
│   ├── 03_train.ipynb            # Model training
│   └── 04_inference.ipynb        # Inference & submission packaging
│
├── shp-file/                     # Sample shapefiles (Bridge, Road, Water, etc.)
├── tiles/                        # Generated tiles (on Google Drive)
│   ├── images/                   # 8,449 RGB tile arrays (.npy, 512×512×3)
│   ├── masks/                    # 8,449 segmentation masks (.npy, 512×512)
│   └── dataset_meta.json         # Tile metadata (village, class counts)
│
├── checkpoints/                  # Saved model weights (.pth)
├── logs/                         # Training history (JSON)
└── outputs/                      # Inference results (GeoTIFF, GPKG, PNG)
```

---

## 5. Detailed Module Descriptions

### 5.1 `src/config.py` — Configuration Hub
- **Single source of truth** for all project parameters
- Auto-detects Colab vs local environment
- Defines all 7 segmentation classes and 4 rooftop types
- Contains `TileConfig`, `TrainConfig`, `InferConfig`, `EvalConfig` dataclasses
- Maps shapefile names → class IDs (handles both CG and PB naming)
- Class colors for visualization: Building=Red, Road=Yellow, Water=Blue, Utility=Green, Bridge=Orange, Railway=Purple

### 5.2 `src/data_pipeline.py` — Tiling Engine
- **Input:** Full-village orthophoto (GeoTIFF, ~10K×10K pixels) + shapefiles
- **Output:** 512×512 pixel tile pairs (image.npy + mask.npy)
- **Key Operations:**
  - Loads and reprojects shapefiles to orthophoto CRS
  - Rasterizes vector labels onto pixel grid (priority-ordered: building > road > water > utility > bridge > railway)
  - Buffers line features (1.5m) and point features (3.0m) to create polygon masks
  - Sliding window with 128px overlap, discards tiles with <1% labeled pixels
  - Generates `dataset_meta.json` with per-tile village name and class pixel counts
  - Village-based train/val split (holds out 1 entire village for validation)

### 5.3 `src/augmentations.py` — Data Augmentation
- **Spatial Augmentations:** HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, Affine (shear)
- **Color Augmentations:** RandomBrightnessContrast, HueSaturationValue, GaussNoise
- **Regularization:** GaussianBlur, MedianBlur, CoarseDropout (cutout)
- **Advanced:** Mosaic augmentation (4 tiles combined), Copy-Paste for rare classes
- Built with **Albumentations v2** API (compatible with latest library)

### 5.4 `src/model.py` — Model Architecture

#### Primary Model: SegFormer (mit_b3)
- **Architecture:** Hierarchical Vision Transformer encoder + lightweight MLP decoder
- **Reference:** Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", NeurIPS 2021
- **Backbone:** Mix Transformer B3 (MiT-B3) — 4-stage hierarchical encoder
- **Pretrained:** `nvidia/segformer-b3-finetuned-ade-512-512` from HuggingFace
- **Parameters:** 47.2 million (all trainable)
- **Input:** (B, 3, 512, 512) RGB images
- **Output:** (B, 7, 512, 512) per-pixel class logits
- **Secondary Head:** Rooftop type classification (RCC, Tiled, Tin, Thatched)

#### Why SegFormer?
1. **Best accuracy/speed tradeoff** for 512×512 dense prediction
2. Hierarchical transformer captures both local texture and global context
3. No positional encoding → handles variable resolutions
4. Proven on ADE20K (150 classes) → transfers well to our 7-class task
5. Lightweight MLP decoder avoids heavy computation

#### Alternative Architectures (implemented but not primary):
- **DeepLabV3+** with ResNet-101 backbone
- **Swin-UNet** with Swin Transformer backbone

### 5.5 `src/losses.py` — Loss Function

#### Boundary-Aware Dice + Focal Loss (`boundary_dice_focal`)
A composite loss function specifically designed for extreme class imbalance:

```
L_total = λ_dice · L_dice + λ_focal · L_focal + λ_boundary · L_boundary
        = 1.0   · Dice   + 1.0    · Focal   + 0.5       · Boundary
```

| Component | Purpose | How it helps |
|-----------|---------|-------------|
| **Dice Loss** | Region overlap | Handles class imbalance (background dominance) |
| **Focal Loss** | Hard pixel mining | Focuses on misclassified pixels (γ=2.0) |
| **Boundary Loss** | Edge quality | Penalizes errors at feature boundaries |

- **Class Weights:** `[0.1, 2.0, 1.5, 2.0, 3.0, 3.0, 3.0]` — downweights background, boosts rare classes

### 5.6 `src/train.py` — Training Engine

| Feature | Implementation |
|---------|---------------|
| **Optimizer** | AdamW (weight_decay=0.01) |
| **Scheduler** | Cosine Annealing with warm restarts |
| **Warmup** | Linear warmup for 5 epochs (LR: 0.01x → 1x) |
| **Mixed Precision** | PyTorch AMP (autocast + GradScaler) — 2x faster, 40% less memory |
| **EMA** | Exponential Moving Average (decay=0.9999) — smoother, more stable model |
| **Gradient Clipping** | max_norm=1.0 — prevents exploding gradients |
| **Early Stopping** | patience=15 epochs on val mIoU |
| **Checkpointing** | Saves best.pth (highest val mIoU) + periodic + final |
| **Auto-backup** | Checkpoints auto-saved to Google Drive |

**Training Data Split:** Village-based split (not random). 5 villages for training, 1 village (nagul_madase_ghotpal, 552 tiles) held out for validation. This ensures the model generalizes to unseen villages.

### 5.7 `src/inference.py` — Inference Engine
- **Sliding Window:** 512×512 tiles with 256px overlap for smooth stitching
- **Test-Time Augmentation (TTA):** Horizontal flip + vertical flip + 90° rotation → averages predictions from multiple views
- **Ensemble Support:** Can combine predictions from SegFormer + DeepLabV3+ + Swin-UNet (weighted average)
- **Output:** Full-resolution segmentation map as GeoTIFF (preserves geospatial metadata)

### 5.8 `src/postprocess.py` — Post-Processing
- **Morphological Cleanup:** Opening (remove noise) + Closing (fill holes) per class
- **CRF Refinement:** Conditional Random Field to sharpen boundaries
- **Vectorization:** Raster predictions → vector polygons (GeoPackage/GPKG)
- **Size Filtering:** Remove buildings < 4 sqm, road segments < 5m
- **Simplification:** Douglas-Peucker algorithm (0.5m tolerance) for clean polygons
- **Export:** Cloud Optimized GeoTIFF (COG) + GPKG

### 5.9 `src/evaluate.py` — Evaluation Metrics
- **Per-class IoU** (Intersection over Union)
- **Mean IoU** (mIoU) — primary competition metric
- **Per-class F1 Score**
- **Mean F1** (mF1)
- **Boundary IoU** — measures accuracy at feature boundaries
- **Confusion Matrix** — full NxN class-level analysis
- **Overall Pixel Accuracy** (OA)

---

## 6. Training Configuration (Final)

```python
TrainConfig(
    model_name      = 'segformer_simple',
    backbone        = 'mit_b3',
    pretrained      = True,            # ImageNet + ADE20K pretrained
    num_classes     = 7,
    
    epochs          = 80,
    batch_size      = 4,               # Fits T4 GPU (15 GB VRAM)
    learning_rate   = 3e-5,
    weight_decay    = 0.01,
    warmup_epochs   = 5,
    min_lr          = 1e-6,
    
    optimizer       = 'adamw',
    scheduler       = 'cosine',
    
    loss_fn         = 'boundary_dice_focal',
    class_weights   = [0.1, 2.0, 1.5, 2.0, 3.0, 3.0, 3.0],
    
    use_amp         = True,            # Mixed precision
    use_ema         = True,            # Exponential Moving Average
    ema_decay       = 0.9999,
    
    use_augmentation = True,
    use_mosaic       = True,
    
    early_stopping_patience = 15,
    save_every_n_epochs     = 5,
    seed                    = 42,
)
```

---

## 7. Execution Platform

| Resource | Specification |
|----------|--------------|
| **Platform** | Google Colab (free tier) |
| **GPU** | NVIDIA Tesla T4 (15.6 GB VRAM) |
| **RAM** | 12.7 GB system RAM |
| **Storage** | Google Drive (persistent) + Colab local SSD (fast, ephemeral) |
| **Framework** | PyTorch 2.x + HuggingFace Transformers |
| **Python** | 3.12 |
| **Local Dev** | macOS (Apple Silicon) — code editing only, no training |

---

## 8. Submission Pipeline Summary

### Step 1: NB01 — Data Preparation
- Downloaded 4 zip files from hackathon portal to Google Drive
- Extracted orthophotos and shapefiles for 6 training villages (CG state)
- Built `data_manifest.json` — inventory of all orthophotos and their matched shapefiles
- Verified CRS consistency (all EPSG:32644 — UTM Zone 44N)
- Identified shapefile naming differences between CG and PB formats

### Step 2: NB02 — Tiling
- Processed all 6 village orthophotos through the tiling pipeline
- Generated **8,449 tile pairs** (image + mask), each 512×512 pixels
- Tiles saved as `.npy` (NumPy arrays) for fast I/O during training
- Applied 128px overlap between adjacent tiles
- Filtered out tiles with <1% labeled pixels (background-only tiles discarded)
- Generated `dataset_meta.json` (1.9 MB) — per-tile metadata including village name and class pixel counts
- Backed up all tiles to Google Drive (8,449 images + 8,449 masks)
- **Output:** `/content/drive/MyDrive/IITT_AIML/tiles/` (images/, masks/, dataset_meta.json)

### Step 3: NB03 — Training
- Model training executed on Google Colab (Tesla T4) using SegFormer (MiT-B3)
- Training setup includes mixed precision (AMP), EMA, cosine scheduler with warmup, and boundary-aware Dice+Focal loss
- Runtime optimization uses local SSD tile cache to avoid Drive FUSE bottlenecks
- Memory-safe configuration uses batch size tuning and CUDA memory management for stable execution
- **Training outputs:** `best.pth`, `final.pth`, periodic `epoch_*.pth`, and `training_history.json`

### Step 4: NB04 — Inference & Submission
- Load best checkpoint from training
- Download and process test orthophotos
- Run sliding window inference with TTA on test images
- Apply post-processing (morphological cleanup, vectorization)
- Generate COG (Cloud Optimized GeoTIFF) and GPKG outputs
- Package submission zip
- **Submission artifacts:** GeoTIFF predictions, confidence maps, vector outputs, and packaged submission zip

---

## 9. Key Technical Decisions & Justifications

| Decision | Why |
|----------|-----|
| **SegFormer over U-Net** | Transformer captures global context (buildings relate to roads); lightweight MLP decoder is efficient on T4 |
| **MiT-B3 backbone** | Best balance of accuracy/speed. B4/B5 are too large for T4; B0/B1 are too small for 7-class task |
| **Village-based split** | Random pixel split causes data leakage (adjacent tiles overlap). Village split tests true generalization |
| **Boundary loss** | Feature boundaries (building edges, road borders) are critical for mapping accuracy |
| **EMA** | Averaged model weights are more robust, reduce overfitting on small dataset |
| **Cosine scheduler with warmup** | Prevents early overfitting; warmup stabilizes initial training with pretrained weights |
| **AMP** | Halves memory usage, doubles throughput — essential for T4 with 47M params |
| **.npy tile format** | Fastest I/O for training. No JPEG/PNG decode overhead per batch |
| **Copy-paste augmentation** | Artificially increases rare class frequency (utility, bridge) |

---

## 10. Challenges Faced & Solutions

| Challenge | Solution |
|-----------|----------|
| **Extreme class imbalance** (bg >90%) | Class weights [0.1, 2.0, 1.5, 2.0, 3.0, 3.0, 3.0] + Focal Loss (γ=2.0) |
| **Railway class empty** (0 samples in CG) | Model trained on 6 active classes; railway handled gracefully via weighted loss |
| **Bridge class very rare** (20 tiles / 0.2%) | Copy-paste augmentation + class weight 3.0 + Dice loss per-class |
| **Albumentations v2 breaking changes** | Runtime monkey-patch in NB03 Cell 2 fixes deprecated API |
| **Drive FUSE slow I/O** (37 min/epoch) | One-time copy to local SSD (5 min) → epochs drop to ~5-8 min |
| **OOM on T4 with batch 8** | Reduced to batch 4, halved LR proportionally, added CUDA cache clearing |
| **Large orthophotos** (10K×10K px) | Tiling with sliding window (512×512, stride 384) |
| **Line/point features in shapefiles** | Buffered to polygons (lines 1.5m, points 3.0m) before rasterization |
| **CG vs PB shapefile naming** | Alias mapping in config (e.g., `Built_Up_Area_typ` → `Built_Up_Area_type`) |

---

## 11. Submission Outputs

### Training Artifacts (NB03):
- `best.pth` — Best model checkpoint (highest validation mIoU)
- `final.pth` — Last epoch model
- `training_history.json` — Full epoch-by-epoch metrics
- Training curves plot (loss, mIoU, F1 over epochs)

### Inference Artifacts (NB04):
- `*_segmentation.tif` — Full-resolution segmentation maps (Cloud Optimized GeoTIFF)
- `*_confidence.tif` — Per-pixel confidence maps
- `*.gpkg` — Vectorized predictions (GeoPackage — buildings as polygons, roads as lines)
- `*_prediction_viz.png` — Colored visualization images
- `submission.zip` — Packaged deliverable for hackathon submission

---

## 12. Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **mIoU** | Mean Intersection over Union (all 7 classes) | > 0.65 |
| **mF1** | Mean F1 Score (harmonic mean of precision & recall) | > 0.70 |
| **Building IoU** | IoU for building class specifically | > 0.75 |
| **Road IoU** | IoU for road class specifically | > 0.60 |
| **Boundary IoU** | Accuracy at feature edges (2-pixel boundary) | > 0.40 |
| **OA** | Overall Pixel Accuracy | > 0.90 |

---

## 13. Technology Stack

| Category | Technology |
|----------|-----------|
| **Deep Learning** | PyTorch 2.x |
| **Model** | HuggingFace Transformers (SegFormer) |
| **Augmentations** | Albumentations v2 |
| **Geospatial** | Rasterio, GeoPandas, Shapely, Fiona |
| **Computation** | Google Colab (T4 GPU) |
| **Storage** | Google Drive |
| **Visualization** | Matplotlib |
| **Format** | NumPy (.npy), GeoTIFF (COG), GeoPackage (GPKG) |

---

## 14. References

1. Xie, E. et al. (2021). "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." NeurIPS 2021.
2. SVAMITVA Scheme — Ministry of Panchayati Raj, Government of India.
3. NVIDIA SegFormer pretrained models — `nvidia/segformer-b3-finetuned-ade-512-512`
4. Albumentations: Fast image augmentation library (Buslaev et al., 2020)
5. Rasterio: Geospatial raster I/O library

---

## 15. Submission Note

This report is prepared as a submission-facing technical document and summarizes the complete end-to-end methodology, architecture, and deliverables for PS-1.

---

*Project by: IIT Tirupati AI/ML Team*
*Hackathon: PS-1 — AI-Based Feature Extraction from Drone Images*
*Date: March 2026*
