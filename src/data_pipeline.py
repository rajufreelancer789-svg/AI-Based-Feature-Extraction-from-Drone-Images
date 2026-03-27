"""
Data Pipeline — Tiling, mask generation, and dataset creation.
Converts large orthophotos + shapefiles into training-ready 512x512 tiles.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from src.config import (
    TileConfig, SEG_CLASSES, SHAPEFILE_CLASS_MAP, ROOF_TYPES,
    TILES_IMAGES, TILES_MASKS, TILES_DIR,
    DRIVE_TILES, IS_COLAB,
    SHAPEFILE_NAME_ALIASES
)


# ============================================================
# SHAPEFILE LOADING & REPROJECTION
# ============================================================

def load_shapefiles(shp_dir: str, target_crs: str) -> Dict[str, gpd.GeoDataFrame]:
    """
    Load all shapefiles from a directory and reproject to target CRS.
    
    Args:
        shp_dir: Directory containing .shp files
        target_crs: Target CRS string (e.g., 'EPSG:32644')
    
    Returns:
        Dictionary mapping shapefile name → GeoDataFrame
    """
    shapefiles = {}
    shp_dir = Path(shp_dir)
    
    for shp_path in sorted(shp_dir.glob('*.shp')):
        name = shp_path.stem  # e.g., 'Built_Up_Area_type'
        try:
            gdf = gpd.read_file(shp_path)
            if len(gdf) == 0:
                print(f'  SKIP (empty): {name}')
                continue
            
            # Drop rows with null or invalid geometries (PB Bridge has some)
            before = len(gdf)
            gdf = gdf[gdf.geometry.notna()].copy()
            gdf = gdf[gdf.geometry.is_valid].copy()
            dropped = before - len(gdf)
            if dropped > 0:
                print(f'  Dropped {dropped} null/invalid geometries from {name}')
            if len(gdf) == 0:
                print(f'  SKIP (all invalid): {name}')
                continue
            
            # Reproject to match orthophoto CRS
            if str(gdf.crs) != target_crs:
                gdf = gdf.to_crs(target_crs)
            
            # Normalize name using aliases (PB → CG naming)
            canonical = SHAPEFILE_NAME_ALIASES.get(name, name)
            if canonical != name:
                print(f'  Alias: {name} → {canonical}')
            
            shapefiles[canonical] = gdf
            print(f'  Loaded: {canonical} → {len(gdf)} features (CRS: {gdf.crs})')
        except Exception as e:
            print(f'  ERROR loading {name}: {e}')
    
    return shapefiles


def buffer_geometries(gdf: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    """Buffer point/line geometries to create polygons for rasterization."""
    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.buffer(buffer_m)
    return gdf


# ============================================================
# MASK GENERATION — Convert shapefiles to raster masks
# ============================================================

def create_segmentation_mask(
    shapefiles: Dict[str, gpd.GeoDataFrame],
    window_bounds: Tuple[float, float, float, float],
    transform: rasterio.transform.Affine,
    shape: Tuple[int, int],
    config: TileConfig
) -> np.ndarray:
    """
    Create a multi-class segmentation mask from shapefiles.
    
    Priority order (higher class index overwrites lower):
    background(0) < road(2) < waterbody(3) < building(1) < utility(4) < bridge(5)
    
    Args:
        shapefiles: Dict of shapefile name → GeoDataFrame
        window_bounds: (left, bottom, right, top) bounds of the tile
        transform: Affine transform for the tile
        shape: (height, width) of the output mask
        config: TileConfig with buffer settings
    
    Returns:
        np.ndarray of shape (H, W) with class indices
    """
    mask = np.zeros(shape, dtype=np.uint8)
    window_box = box(*window_bounds)
    
    # Process shapefiles in priority order (lower priority first)
    priority_order = [
        ('Road', 2), ('Road_Centre_Line', 2),
        ('Water_Body', 3), ('Water_Body_Line', 3), ('Waterbody_Point', 3),
        ('Railway', 6),
        ('Built_Up_Area_type', 1),
        ('Utility_Poly', 4), ('Utility', 4),
        ('Bridge', 5),
    ]
    
    for shp_name, class_id in priority_order:
        if shp_name not in shapefiles:
            continue
        
        gdf = shapefiles[shp_name]
        
        # Clip features to tile bounds
        try:
            clipped = gdf[gdf.intersects(window_box)].copy()
        except Exception:
            continue
        
        if len(clipped) == 0:
            continue
        
        # Buffer line/point features to create polygons
        geom_type = clipped.geom_type.iloc[0]
        if geom_type in ('LineString', 'MultiLineString'):
            clipped = buffer_geometries(clipped, config.line_buffer_m)
        elif geom_type in ('Point', 'MultiPoint'):
            clipped = buffer_geometries(clipped, config.point_buffer_m)
        
        # Rasterize
        try:
            shapes = [(geom, class_id) for geom in clipped.geometry if geom is not None and geom.is_valid]
            if shapes:
                layer = rasterize(
                    shapes,
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype='uint8'
                )
                mask = np.where(layer > 0, layer, mask)
        except Exception as e:
            print(f'    Warning: rasterize failed for {shp_name}: {e}')
    
    return mask


def create_roof_type_mask(
    shapefiles: Dict[str, gpd.GeoDataFrame],
    window_bounds: Tuple[float, float, float, float],
    transform: rasterio.transform.Affine,
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Create a rooftop type mask (only for building pixels).
    
    Returns:
        np.ndarray of shape (H, W) with roof type indices (1-4), 0 = non-building
    """
    mask = np.zeros(shape, dtype=np.uint8)
    window_box = box(*window_bounds)
    
    if 'Built_Up_Area_type' not in shapefiles:
        return mask
    
    gdf = shapefiles['Built_Up_Area_type']
    
    if 'Roof_type' not in gdf.columns:
        return mask
    
    try:
        clipped = gdf[gdf.intersects(window_box)].copy()
    except Exception:
        return mask
    
    if len(clipped) == 0:
        return mask
    
    # Rasterize with roof type as the burn value
    shapes = []
    for _, row in clipped.iterrows():
        geom = row.geometry
        roof_type = row.get('Roof_type', 0)
        if geom is not None and geom.is_valid and roof_type in ROOF_TYPES:
            shapes.append((geom, int(roof_type)))
    
    if shapes:
        mask = rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype='uint8'
        )
    
    return mask


# ============================================================
# TILING — Split large orthophotos into training patches
# ============================================================

def get_tile_windows(
    width: int, height: int, config: TileConfig
) -> List[Tuple[int, int, Window]]:
    """
    Generate sliding window positions for tiling.
    
    Returns:
        List of (col_off, row_off, Window) tuples
    """
    windows = []
    stride = config.stride
    
    for row_off in range(0, height, stride):
        for col_off in range(0, width, stride):
            # Handle edge tiles (pad to full size)
            h = min(config.tile_size, height - row_off)
            w = min(config.tile_size, width - col_off)
            
            # Skip tiles that are too small
            if h < config.tile_size // 2 or w < config.tile_size // 2:
                continue
            
            window = Window(col_off, row_off, w, h)
            windows.append((col_off, row_off, window))
    
    return windows


def _compute_feature_bounds(shapefiles: Dict[str, gpd.GeoDataFrame]):
    """Compute union of all shapefile bounds for spatial pre-filtering."""
    all_bounds = []
    for name, gdf in shapefiles.items():
        if len(gdf) > 0:
            all_bounds.append(gdf.total_bounds)  # (minx, miny, maxx, maxy)
    if not all_bounds:
        return None
    bounds_arr = np.array(all_bounds)
    return (
        bounds_arr[:, 0].min(),  # minx
        bounds_arr[:, 1].min(),  # miny
        bounds_arr[:, 2].max(),  # maxx
        bounds_arr[:, 3].max(),  # maxy
    )


def tile_orthophoto(
    ortho_path: str,
    shapefiles: Dict[str, gpd.GeoDataFrame],
    config: TileConfig,
    village_name: str,
    output_base: str = TILES_DIR,
) -> List[Dict]:
    """
    Tile an entire orthophoto into training patches with corresponding masks.
    
    Optimized: opens TIFF once, pre-filters tiles by shapefile bounds.
    
    Args:
        ortho_path: Path to the orthophoto GeoTIFF
        shapefiles: Dict of loaded/reprojected shapefiles
        config: TileConfig settings
        village_name: Name identifier for this village
        output_base: Base output directory
    
    Returns:
        List of tile metadata dicts
    """
    output_images = os.path.join(output_base, 'images')
    output_masks = os.path.join(output_base, 'masks')
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_masks, exist_ok=True)
    
    print(f'\nTiling: {os.path.basename(ortho_path)}')
    
    with rasterio.open(ortho_path) as src:
        width, height = src.width, src.height
        ortho_transform = src.transform
        ortho_crs = str(src.crs)
        print(f'  Size: {width}x{height}, CRS: {src.crs}, Res: {src.res[0]*100:.1f} cm/px')
        
        # Generate all windows
        windows = get_tile_windows(width, height, config)
        total_windows = len(windows)
        print(f'  Total windows: {total_windows}')
        
        # ── Spatial pre-filter: only process tiles overlapping shapefile extent ──
        feat_bounds = _compute_feature_bounds(shapefiles)
        if feat_bounds is not None:
            fminx, fminy, fmaxx, fmaxy = feat_bounds
            # Add generous buffer (2 tiles worth)
            buf = config.tile_size * abs(src.res[0]) * 2
            fminx -= buf; fminy -= buf; fmaxx += buf; fmaxy += buf
            
            filtered = []
            for col_off, row_off, win in windows:
                # Get tile bounds in CRS coords
                tb = rasterio.windows.bounds(
                    Window(col_off, row_off, config.tile_size, config.tile_size),
                    ortho_transform
                )
                # tb = (left, bottom, right, top)
                if tb[2] < fminx or tb[0] > fmaxx or tb[3] < fminy or tb[1] > fmaxy:
                    continue  # No overlap
                filtered.append((col_off, row_off, win))
            
            print(f'  After spatial filter: {len(filtered)} windows '
                  f'(skipped {total_windows - len(filtered)} outside feature bounds)')
            windows = filtered
        
        # ── Process tiles (file stays open) ──
        tiles_meta = []
        saved = 0
        skipped = 0
        
        for i, (col_off, row_off, _) in enumerate(windows):
            tile_id = f'{village_name}_{col_off}_{row_off}'
            
            try:
                # Read window
                window = Window(col_off, row_off, config.tile_size, config.tile_size)
                window = window.intersection(Window(0, 0, src.width, src.height))
                
                tile_data = src.read(config.bands, window=window)
                
                # Pad edge tiles
                if tile_data.shape[1] < config.tile_size or tile_data.shape[2] < config.tile_size:
                    padded = np.zeros(
                        (len(config.bands), config.tile_size, config.tile_size),
                        dtype=tile_data.dtype
                    )
                    padded[:, :tile_data.shape[1], :tile_data.shape[2]] = tile_data
                    tile_data = padded
                
                # Skip all-black tiles (nodata / outside coverage)
                if tile_data.max() == 0:
                    skipped += 1
                    continue
                
                tile_transform = src.window_transform(window)
                tile_bounds = rasterio.windows.bounds(window, ortho_transform)
                
                # Create masks
                seg_mask = create_segmentation_mask(
                    shapefiles, tile_bounds, tile_transform,
                    (config.tile_size, config.tile_size), config
                )
                
                roof_mask = create_roof_type_mask(
                    shapefiles, tile_bounds, tile_transform,
                    (config.tile_size, config.tile_size)
                )
                
                # Skip tiles with too few labeled pixels
                labeled_ratio = np.count_nonzero(seg_mask) / seg_mask.size
                if labeled_ratio < config.min_labeled_ratio:
                    skipped += 1
                    continue
                
                # Save image
                np.save(os.path.join(output_images, f'{tile_id}.npy'), tile_data)
                
                # Save masks (seg + roof stacked)
                mask_stack = np.stack([seg_mask, roof_mask], axis=0)  # (2, H, W)
                np.save(os.path.join(output_masks, f'{tile_id}.npy'), mask_stack)
                
                # Compute stats
                unique, counts = np.unique(seg_mask, return_counts=True)
                class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
                
                tiles_meta.append({
                    'tile_id': tile_id,
                    'ortho': os.path.basename(ortho_path),
                    'col_off': col_off,
                    'row_off': row_off,
                    'bounds': list(tile_bounds),
                    'crs': ortho_crs,
                    'labeled_ratio': float(labeled_ratio),
                    'class_distribution': class_dist,
                    'has_buildings': bool(1 in class_dist),
                    'has_roads': bool(2 in class_dist),
                    'has_water': bool(3 in class_dist),
                    'has_utility': bool(4 in class_dist),
                    'has_bridge': bool(5 in class_dist),
                    'has_railway': bool(6 in class_dist),
                })
                saved += 1
            except Exception as e:
                print(f'    Error processing tile {tile_id}: {e}')
                skipped += 1
            
            if (i + 1) % 500 == 0:
                print(f'  Progress: {i+1}/{len(windows)} ({saved} saved, {skipped} skipped)')
    
    print(f'  Done: {saved} tiles saved, {skipped} skipped')
    
    return tiles_meta


# ============================================================
# DATASET CREATION — Combine all villages into one dataset
# ============================================================

def create_dataset(
    ortho_paths: List[str],
    shp_dirs: List[str],
    config: TileConfig = TileConfig(),
    output_base: str = TILES_DIR,
) -> Dict:
    """
    Create the complete tiled dataset from all orthophotos.
    
    Args:
        ortho_paths: List of orthophoto file paths
        shp_dirs: List of shapefile directories (matched to ortho_paths)
        config: TileConfig settings
        output_base: Base output directory
    
    Returns:
        Dataset metadata dict
    """
    all_tiles = []
    
    for ortho_path, shp_dir in zip(ortho_paths, shp_dirs):
        # Get orthophoto CRS
        with rasterio.open(ortho_path) as src:
            target_crs = str(src.crs)
        
        # Load and reproject shapefiles
        print(f'\nLoading shapefiles from: {shp_dir}')
        shapefiles = load_shapefiles(shp_dir, target_crs)
        
        # Extract village name from filename (handles multi-village + spaces)
        fname_stem = os.path.splitext(os.path.basename(ortho_path))[0]
        parts = fname_stem.replace(' ', '_').replace('-', '_').split('_')
        name_parts = [p for p in parts if not p.isdigit()
                      and p.upper() not in ('ORTHO', 'ORI', '3857')]
        village_name = '_'.join(name_parts).lower() if name_parts else fname_stem.lower()
        
        # Tile this orthophoto
        tiles = tile_orthophoto(
            ortho_path, shapefiles, config, village_name, output_base
        )
        all_tiles.extend(tiles)
    
    # Save metadata
    metadata = {
        'total_tiles': len(all_tiles),
        'tile_size': config.tile_size,
        'overlap': config.overlap,
        'tiles': all_tiles,
        'class_summary': _compute_class_summary(all_tiles),
    }
    
    meta_path = os.path.join(output_base, 'dataset_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also save to Drive if on Colab
    if IS_COLAB and DRIVE_TILES:
        os.makedirs(DRIVE_TILES, exist_ok=True)
        drive_meta = os.path.join(DRIVE_TILES, 'dataset_meta.json')
        with open(drive_meta, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f'\n{"="*60}')
    print(f'DATASET CREATED')
    print(f'{"="*60}')
    print(f'  Total tiles: {len(all_tiles)}')
    print(f'  Tile size: {config.tile_size}x{config.tile_size}')
    print(f'  Saved to: {output_base}')
    print(f'\nClass distribution:')
    for cls_name, count in metadata['class_summary'].items():
        print(f'  {cls_name}: {count} tiles')
    
    return metadata


def _compute_class_summary(tiles: List[Dict]) -> Dict[str, int]:
    """Count how many tiles contain each class."""
    summary = {
        'building': sum(1 for t in tiles if t.get('has_buildings')),
        'road': sum(1 for t in tiles if t.get('has_roads')),
        'waterbody': sum(1 for t in tiles if t.get('has_water')),
        'utility': sum(1 for t in tiles if t.get('has_utility')),
        'bridge': sum(1 for t in tiles if t.get('has_bridge')),
        'railway': sum(1 for t in tiles if t.get('has_railway')),
    }
    return summary


# ============================================================
# TORCH DATASET — For training
# ============================================================

def get_train_val_split(
    metadata_path: str,
    train_ratio: float = 0.85,
    seed: int = 42,
    strategy: str = 'random'  # 'random' or 'village_split'
) -> Tuple[List[str], List[str]]:
    """
    Split tiles into train/val sets.
    
    Args:
        metadata_path: Path to dataset_meta.json
        train_ratio: Fraction for training
        seed: Random seed
        strategy: 'random' or 'village_split' (spatial cross-validation)
    
    Returns:
        (train_tile_ids, val_tile_ids)
    """
    with open(metadata_path) as f:
        meta = json.load(f)
    
    tiles = meta['tiles']
    
    if strategy == 'village_split':
        # Group by village → leave one out for validation
        villages = {}
        for t in tiles:
            village = t['tile_id'].rsplit('_', 2)[0]
            villages.setdefault(village, []).append(t['tile_id'])
        
        village_names = sorted(villages.keys())
        np.random.seed(seed)
        val_village = np.random.choice(village_names)
        
        val_ids = villages[val_village]
        train_ids = [tid for v, tids in villages.items() if v != val_village for tid in tids]
        
        print(f'Village split: val={val_village} ({len(val_ids)} tiles)')
        print(f'  Train: {len(train_ids)} tiles from {len(village_names)-1} villages')
    else:
        # Random split
        np.random.seed(seed)
        tile_ids = [t['tile_id'] for t in tiles]
        np.random.shuffle(tile_ids)
        
        split_idx = int(len(tile_ids) * train_ratio)
        train_ids = tile_ids[:split_idx]
        val_ids = tile_ids[split_idx:]
    
    return train_ids, val_ids
