# AI-Based Feature Extraction from Drone Images

![QC Output Preview](assets/qc_output.png)

SVAMITVA Scheme | IIT Tirupati AI/ML Hackathon | PS-1 Submission

This repository contains an end-to-end geospatial AI pipeline for extracting village-scale map features from drone orthophotos using semantic segmentation.

## Problem
Build an automated system that detects and segments key features from orthophotos:
- Building
- Road
- Waterbody
- Utility
- Bridge
- Railway

## Submission Highlights
- Transformer-based segmentation model: SegFormer (MiT-B3)
- 7-class semantic segmentation pipeline
- 8,449 training tiles generated from 6 villages
- Boundary-aware loss design for class imbalance
- Sliding-window inference with TTA for full orthophoto prediction
- Geospatial deliverables: COG raster + GPKG vector outputs

## Repository Structure
- src: core pipeline modules (config, preprocessing, model, training, inference, postprocess, evaluation)
- notebooks: Colab execution notebooks
- shp-file: shapefile samples
- PROJECT_REPORT.md: full submission technical report

## Tech Stack
- PyTorch 2.x
- HuggingFace Transformers (SegFormer)
- Albumentations v2
- Rasterio, GeoPandas, Shapely
- Google Colab (Tesla T4)

## Pipeline
1. Data setup and manifest creation
2. Orthophoto tiling and mask rasterization
3. Model training (SegFormer + boundary_dice_focal)
4. Inference (sliding window + TTA)
5. Post-processing and vectorization
6. Submission packaging

## Note on QC Preview Image
Place your final QC preview image at assets/qc_output.png so it renders as the top banner in this README.
