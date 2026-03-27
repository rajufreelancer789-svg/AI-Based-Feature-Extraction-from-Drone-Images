"""
Post-Processing — CRF, morphological ops, vectorization, COG/GPKG export.
Converts raw predictions into competition-ready deliverables.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rasterio
from rasterio.transform import from_bounds
from rasterio.features import shapes as rasterio_shapes
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from src.config import (
    InferConfig, SEG_CLASSES, ROOF_TYPES, SEG_COLORS,
    OUTPUT_DIR, DRIVE_OUTPUTS, IS_COLAB,
)


# ============================================================
# MORPHOLOGICAL REFINEMENT
# ============================================================

def morphological_cleanup(
    class_map: np.ndarray,
    config: InferConfig = InferConfig(),
) -> np.ndarray:
    """
    Apply morphological operations to clean predictions.
    
    Operations:
    1. Remove small noise (opening)
    2. Fill small holes (closing)
    3. Smooth boundaries
    """
    from scipy import ndimage
    
    cleaned = class_map.copy()
    
    for cls_id in range(1, len(SEG_CLASSES)):
        binary = (cleaned == cls_id).astype(np.uint8)
        
        if binary.sum() == 0:
            continue
        
        # Determine struct element size based on class
        if cls_id == 1:  # Building
            kernel_size = 5
        elif cls_id == 2:  # Road
            kernel_size = 3
        elif cls_id == 3:  # Waterbody
            kernel_size = 7
        else:
            kernel_size = 3
        
        struct = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct, kernel_size // 2)
        
        # Opening: remove small noise
        binary = ndimage.binary_opening(binary, structure=struct, iterations=1)
        
        # Closing: fill small holes
        binary = ndimage.binary_closing(binary, structure=struct, iterations=1)
        
        # Remove small connected components
        labeled, num_features = ndimage.label(binary)
        if num_features > 0:
            component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            # Minimum size threshold (in pixels)
            min_pixels = {
                1: 50,   # Buildings: ~0.06 sqm * 50 ≈ 3 sqm
                2: 30,   # Roads: smaller threshold
                3: 100,  # Water bodies: larger threshold
                4: 10,   # Utilities: allow small
                5: 20,   # Bridges
                6: 30,   # Railway
            }.get(cls_id, 20)
            
            for comp_id in range(1, num_features + 1):
                if component_sizes[comp_id - 1] < min_pixels:
                    binary[labeled == comp_id] = 0
        
        # Apply cleaned mask back
        # First remove old class pixels, then add cleaned
        cleaned[cleaned == cls_id] = 0   # only clear pixels currently assigned to this class
        cleaned[binary > 0] = cls_id
    
    return cleaned


# ============================================================
# VECTORIZATION — Raster to Vector (GPKG)
# ============================================================

def vectorize_predictions(
    class_map: np.ndarray,
    transform: rasterio.transform.Affine,
    crs: str,
    config: InferConfig = InferConfig(),
    roof_map: Optional[np.ndarray] = None,
    confidence_map: Optional[np.ndarray] = None,
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Convert raster predictions to vector polygons.
    
    Returns:
        Dictionary mapping class name → GeoDataFrame with polygons
    """
    results = {}
    
    for cls_id, cls_name in SEG_CLASSES.items():
        if cls_id == 0:  # Skip background
            continue
        
        binary = (class_map == cls_id).astype(np.uint8)
        
        if binary.sum() == 0:
            continue
        
        # Extract polygons using rasterio
        polygons = []
        values = []
        
        for geom, val in rasterio_shapes(binary, mask=binary > 0, transform=transform):
            if val == 0:
                continue
            poly = shape(geom)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_empty or poly.area == 0:
                continue
            polygons.append(poly)
            values.append(val)
        
        if not polygons:
            continue
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'class_id': cls_id, 'class_name': cls_name, 'geometry': polygons},
            crs=crs,
        )
        
        # Simplify geometry (Douglas-Peucker)
        if config.simplify_tolerance > 0:
            gdf['geometry'] = gdf.geometry.simplify(
                config.simplify_tolerance, preserve_topology=True
            )
        
        # Filter by minimum area
        if cls_id == 1:  # Building
            min_area = config.min_building_area_sqm
            gdf = gdf[gdf.geometry.area >= min_area].copy()
        
        # Add area column
        gdf['area_sqm'] = gdf.geometry.area
        
        # Add rooftop type for buildings
        if cls_id == 1 and roof_map is not None:
            gdf = _assign_roof_types(gdf, roof_map, transform)
        
        # Add confidence (mean confidence within polygon)
        if confidence_map is not None:
            gdf['confidence'] = gdf.apply(
                lambda row: _mean_confidence_in_polygon(
                    row.geometry, confidence_map, transform
                ), axis=1
            )
        
        results[cls_name] = gdf
        print(f'  {cls_name}: {len(gdf)} polygons')
    
    return results


def _assign_roof_types(
    gdf: gpd.GeoDataFrame,
    roof_map: np.ndarray,
    transform: rasterio.transform.Affine,
) -> gpd.GeoDataFrame:
    """Assign rooftop type to each building polygon using majority voting."""
    gdf = gdf.copy()
    roof_types = []
    
    for _, row in gdf.iterrows():
        poly = row.geometry
        # Get bounding box in pixel coords
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        
        # Convert to pixel coordinates
        inv_transform = ~transform
        col_min, row_max = inv_transform * (bounds[0], bounds[1])
        col_max, row_min = inv_transform * (bounds[2], bounds[3])
        
        r0 = max(0, int(row_min))
        r1 = min(roof_map.shape[0], int(row_max) + 1)
        c0 = max(0, int(col_min))
        c1 = min(roof_map.shape[1], int(col_max) + 1)
        
        if r1 <= r0 or c1 <= c0:
            roof_types.append(0)
            continue
        
        # Get roof type in region
        region = roof_map[r0:r1, c0:c1]
        valid = region[region > 0]
        
        if len(valid) == 0:
            roof_types.append(0)
        else:
            # Majority vote
            unique, counts = np.unique(valid, return_counts=True)
            roof_types.append(int(unique[counts.argmax()]))
    
    gdf['roof_type_id'] = roof_types
    gdf['roof_type'] = gdf['roof_type_id'].map(
        lambda x: ROOF_TYPES.get(x, 'Unknown')
    )
    
    return gdf


def _mean_confidence_in_polygon(
    polygon, confidence_map: np.ndarray,
    transform: rasterio.transform.Affine,
) -> float:
    """Get mean confidence score within a polygon."""
    try:
        bounds = polygon.bounds
        inv_transform = ~transform
        col_min, row_max = inv_transform * (bounds[0], bounds[1])
        col_max, row_min = inv_transform * (bounds[2], bounds[3])
        
        r0 = max(0, int(row_min))
        r1 = min(confidence_map.shape[0], int(row_max) + 1)
        c0 = max(0, int(col_min))
        c1 = min(confidence_map.shape[1], int(col_max) + 1)
        
        if r1 <= r0 or c1 <= c0:
            return 0.0
        
        region = confidence_map[r0:r1, c0:c1]
        return float(np.mean(region))
    except Exception:
        return 0.0


# ============================================================
# EXPORT — COG and GPKG
# ============================================================

def save_as_cog(
    data: np.ndarray,
    profile: dict,
    output_path: str,
    nodata: int = 255,
    dtype: str = 'uint8',
):
    """
    Save raster data as a Cloud Optimized GeoTIFF (COG).
    
    COG = GeoTIFF with internal tiling + overviews.
    Required output format per IIT Tirupati/OGC specs.
    """
    profile = profile.copy()
    profile.update({
        'driver': 'GTiff',
        'dtype': dtype,
        'count': 1,
        'nodata': nodata,
        'compress': 'deflate',
        'predictor': 2,
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
    })
    
    # Handle 2D arrays
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    
    profile['height'] = data.shape[1]
    profile['width'] = data.shape[2]
    profile['count'] = data.shape[0]
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(dtype))
        
        # Add overviews for COG (pyramids)
        overview_levels = [2, 4, 8, 16]
        if max(data.shape[1], data.shape[2]) > 4096:
            overview_levels.extend([32, 64])
        
        dst.build_overviews(overview_levels, rasterio.enums.Resampling.nearest)
        dst.update_tags(ns='rio_overview', resampling='nearest')
    
    print(f'  COG saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)')


def save_as_gpkg(
    vectors: Dict[str, gpd.GeoDataFrame],
    output_path: str,
    target_crs: str = None,
):
    """
    Save vectorized predictions as GeoPackage (GPKG).
    
    Each class becomes a separate layer in the GPKG.
    Required output format per IIT Tirupati/OGC specs.
    """
    if not vectors:
        print('  No vectors to save.')
        return
    
    for cls_name, gdf in vectors.items():
        if target_crs and str(gdf.crs) != target_crs:
            gdf = gdf.to_crs(target_crs)
        
        layer_name = cls_name.lower().replace(' ', '_')
        gdf.to_file(output_path, layer=layer_name, driver='GPKG')
    
    print(f'  GPKG saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)')


# ============================================================
# VISUALIZATION
# ============================================================

def colorize_prediction(class_map: np.ndarray) -> np.ndarray:
    """
    Convert class index map to RGB visualization.
    
    Args:
        class_map: (H, W) uint8 class indices
    
    Returns:
        (H, W, 3) uint8 RGB image
    """
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_id, color in SEG_COLORS.items():
        mask = class_map == cls_id
        rgb[mask] = color
    
    return rgb


def overlay_prediction(
    image: np.ndarray,
    class_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay colored prediction on the original image.
    
    Args:
        image: (H, W, 3) original RGB image
        class_map: (H, W) class indices
        alpha: Overlay transparency
    
    Returns:
        (H, W, 3) blended image
    """
    colored = colorize_prediction(class_map)
    
    # Only overlay non-background pixels
    fg_mask = class_map > 0
    blended = image.copy()
    blended[fg_mask] = (
        (1 - alpha) * image[fg_mask].astype(np.float32) +
        alpha * colored[fg_mask].astype(np.float32)
    ).astype(np.uint8)
    
    return blended


# ============================================================
# FULL POST-PROCESSING PIPELINE
# ============================================================

def postprocess_predictions(
    class_map: np.ndarray,
    ortho_path: str,
    output_dir: str,
    village_name: str,
    config: InferConfig = InferConfig(),
    roof_map: Optional[np.ndarray] = None,
    confidence_map: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """
    Full post-processing pipeline:
    1. Morphological cleanup
    2. Vectorization
    3. Export as COG + GPKG
    
    Args:
        class_map: (H, W) raw prediction
        ortho_path: Path to original orthophoto
        output_dir: Output directory
        village_name: Name for output files
        config: InferConfig settings
    
    Returns:
        Dict with paths to all output files
    """
    print(f'\nPost-processing: {village_name}')
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # 1. Morphological cleanup
    if config.use_morphology:
        print('  Morphological cleanup...')
        class_map = morphological_cleanup(class_map, config)
    
    # 2. Save cleaned raster as COG
    with rasterio.open(ortho_path) as src:
        profile = src.profile.copy()
        crs = str(src.crs)
        transform = src.transform
    
    cog_path = os.path.join(output_dir, f'{village_name}_segmentation.tif')
    save_as_cog(class_map, profile, cog_path)
    paths['segmentation_cog'] = cog_path
    
    # 3. Vectorize
    print('  Vectorizing...')
    vectors = vectorize_predictions(
        class_map, transform, crs, config,
        roof_map=roof_map,
        confidence_map=confidence_map,
    )
    
    # 4. Save as GPKG
    gpkg_path = os.path.join(output_dir, f'{village_name}_features.gpkg')
    save_as_gpkg(vectors, gpkg_path, target_crs=crs)
    paths['features_gpkg'] = gpkg_path
    
    # 5. Save colored visualization
    viz_path = os.path.join(output_dir, f'{village_name}_visualization.tif')
    colored = colorize_prediction(class_map)
    viz_profile = profile.copy()
    viz_profile['count'] = 3
    viz_profile['dtype'] = 'uint8'
    with rasterio.open(viz_path, 'w', **viz_profile) as dst:
        dst.write(np.transpose(colored, (2, 0, 1)))
    paths['visualization'] = viz_path
    
    # 6. Generate summary statistics
    stats = _compute_prediction_stats(class_map, vectors, village_name)
    stats_path = os.path.join(output_dir, f'{village_name}_stats.json')
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    paths['stats'] = stats_path
    
    print(f'  Post-processing complete for {village_name}')
    
    return paths


def _compute_prediction_stats(
    class_map: np.ndarray,
    vectors: Dict[str, gpd.GeoDataFrame],
    village_name: str,
) -> Dict:
    """Compute summary statistics of predictions."""
    total_pixels = class_map.size
    
    stats = {
        'village': village_name,
        'total_pixels': int(total_pixels),
        'classes': {},
    }
    
    for cls_id, cls_name in SEG_CLASSES.items():
        count = int((class_map == cls_id).sum())
        stats['classes'][cls_name] = {
            'pixel_count': count,
            'pixel_fraction': float(count / total_pixels),
        }
        
        if cls_name in vectors:
            gdf = vectors[cls_name]
            stats['classes'][cls_name]['num_polygons'] = len(gdf)
            stats['classes'][cls_name]['total_area_sqm'] = float(gdf.geometry.area.sum())
            
            if cls_name == 'building' and 'roof_type' in gdf.columns:
                roof_dist = gdf['roof_type'].value_counts().to_dict()
                stats['classes'][cls_name]['roof_type_distribution'] = roof_dist
    
    return stats
