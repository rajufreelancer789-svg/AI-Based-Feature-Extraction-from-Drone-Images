"""Deep inspection of all shapefiles + cross-reference with orthophotos."""

import geopandas as gpd
import rasterio
import os
import numpy as np
from pathlib import Path
from shapely.geometry import box

shp_dir = '/Users/appalaraju/IITT_AIML/shp-file'
ortho_dir = '/Users/appalaraju/IITT_AIML/imagery/CG_train'

print("=" * 80)
print("PART 1: DEEP SHAPEFILE INSPECTION")
print("=" * 80)

all_gdfs = {}

for shp_path in sorted(Path(shp_dir).glob('*.shp')):
    name = shp_path.stem
    print(f'\n{"=" * 70}')
    print(f'  {name}')
    print(f'{"=" * 70}')

    gdf = gpd.read_file(shp_path)
    all_gdfs[name] = gdf

    print(f'  Features: {len(gdf)}')
    print(f'  CRS: {gdf.crs}')
    print(f'  Geometry types: {gdf.geom_type.unique().tolist()}')
    print(f'  Columns: {list(gdf.columns)}')

    # Column details
    for col in gdf.columns:
        if col == 'geometry':
            continue
        dtype = gdf[col].dtype
        nunique = gdf[col].nunique()
        nulls = gdf[col].isnull().sum()
        print(f'    {col}: dtype={dtype}, unique={nunique}, nulls={nulls}')
        if nunique <= 20:
            vals = gdf[col].value_counts().to_dict()
            print(f'      Values: {vals}')

    if len(gdf) == 0:
        print('  ** EMPTY - no features **')
        continue

    # Geometry validity
    invalid_count = (~gdf.is_valid).sum()
    empty_count = gdf.is_empty.sum()
    print(f'  Invalid geometries: {invalid_count}')
    print(f'  Empty geometries: {empty_count}')

    # Bounds
    bounds = gdf.total_bounds
    print(f'  Bounds (EPSG:3857):')
    print(f'    X: [{bounds[0]:.2f}, {bounds[2]:.2f}]')
    print(f'    Y: [{bounds[1]:.2f}, {bounds[3]:.2f}]')

    # Area stats for polygons
    geom_type = gdf.geom_type.iloc[0]
    if geom_type in ('Polygon', 'MultiPolygon'):
        areas = gdf.geometry.area
        print(f'  Area (sq units in EPSG:3857):')
        print(f'    min={areas.min():.2f}, max={areas.max():.2f}')
        print(f'    mean={areas.mean():.2f}, median={areas.median():.2f}')
    elif geom_type in ('LineString', 'MultiLineString'):
        lengths = gdf.geometry.length
        print(f'  Length (units in EPSG:3857):')
        print(f'    min={lengths.min():.2f}, max={lengths.max():.2f}')
        print(f'    mean={lengths.mean():.2f}, total={lengths.sum():.2f}')

    # Sample rows
    print(f'  Sample rows:')
    for i, row in gdf.head(3).iterrows():
        row_dict = {c: row[c] for c in gdf.columns if c != 'geometry'}
        gt = row.geometry.geom_type
        row_dict['_geom_type'] = gt
        print(f'    {row_dict}')


# ============================================================
# PART 2: VILLAGE ANALYSIS
# ============================================================
print(f'\n\n{"=" * 80}')
print("PART 2: VILLAGE ANALYSIS (from Built_Up_Area_type)")
print("=" * 80)

buildings = all_gdfs.get('Built_Up_Area_type')
if buildings is not None and 'Village' in buildings.columns:
    villages = buildings['Village'].unique()
    print(f'\nVillages in building data: {villages.tolist()}')
    print(f'\nPer-village breakdown:')
    for v in sorted(villages):
        v_gdf = buildings[buildings['Village'] == v]
        print(f'\n  {v}: {len(v_gdf)} buildings')
        bounds = v_gdf.total_bounds
        print(f'    Bounds X: [{bounds[0]:.2f}, {bounds[2]:.2f}]')
        print(f'    Bounds Y: [{bounds[1]:.2f}, {bounds[3]:.2f}]')
        if 'Roof_type' in v_gdf.columns:
            roof_dist = v_gdf['Roof_type'].value_counts().to_dict()
            print(f'    Roof types: {roof_dist}')
        areas = v_gdf.geometry.area
        print(f'    Building area: min={areas.min():.1f}, max={areas.max():.1f}, mean={areas.mean():.1f}')
else:
    # Try to find village info from other column names
    if buildings is not None:
        print(f'Columns in Built_Up_Area_type: {list(buildings.columns)}')
        # Check if any column has village-like info
        for col in buildings.columns:
            if col == 'geometry':
                continue
            if buildings[col].dtype == 'object':
                print(f'  String column "{col}": {buildings[col].unique().tolist()[:20]}')


# ============================================================
# PART 3: CHECK VILLAGE COVERAGE IN OTHER SHAPEFILES
# ============================================================
print(f'\n\n{"=" * 80}')
print("PART 3: CROSS-SHAPEFILE VILLAGE COVERAGE")
print("=" * 80)

# Check if other shapefiles have Village column
for name, gdf in all_gdfs.items():
    if len(gdf) == 0:
        continue
    village_cols = [c for c in gdf.columns if 'village' in c.lower() or 'vill' in c.lower()]
    if village_cols:
        for vc in village_cols:
            vals = gdf[vc].unique().tolist()
            print(f'  {name}.{vc}: {vals}')
    else:
        # No village column - check spatial overlap with building villages
        print(f'  {name}: No village column. {len(gdf)} features.')


# ============================================================
# PART 4: CROSS-REFERENCE WITH ORTHOPHOTOS
# ============================================================
print(f'\n\n{"=" * 80}')
print("PART 4: ORTHOPHOTO-SHAPEFILE SPATIAL OVERLAP")
print("=" * 80)

# Load orthophoto bounds
ortho_info = []
for root, dirs, files in os.walk(ortho_dir):
    for f in files:
        if f.lower().endswith('.tif'):
            path = os.path.join(root, f)
            try:
                with rasterio.open(path) as src:
                    ortho_info.append({
                        'name': f,
                        'path': path,
                        'crs': str(src.crs),
                        'bounds': src.bounds,
                        'width': src.width,
                        'height': src.height,
                        'res': src.res,
                    })
                    print(f'\n  Orthophoto: {f}')
                    print(f'    CRS: {src.crs}, Size: {src.width}x{src.height}')
                    print(f'    Bounds: {src.bounds}')
                    print(f'    Resolution: {src.res[0]*100:.1f} cm/px')
            except Exception as e:
                print(f'\n  Orthophoto: {f} — ERROR: {e}')

# Reproject shapefile bounds to orthophoto CRS and check overlap
if ortho_info:
    target_crs = ortho_info[0]['crs']  # EPSG:32644
    print(f'\n  Reprojecting shapefiles to {target_crs} for overlap check...')

    for name, gdf in all_gdfs.items():
        if len(gdf) == 0:
            continue
        try:
            gdf_reproj = gdf.to_crs(target_crs)
            shp_bounds = gdf_reproj.total_bounds
            shp_box = box(*shp_bounds)

            print(f'\n  {name} (reprojected bounds):')
            print(f'    X: [{shp_bounds[0]:.2f}, {shp_bounds[2]:.2f}]')
            print(f'    Y: [{shp_bounds[1]:.2f}, {shp_bounds[3]:.2f}]')

            for ortho in ortho_info:
                ob = ortho['bounds']
                ortho_box = box(ob.left, ob.bottom, ob.right, ob.top)
                overlap = shp_box.intersection(ortho_box)
                if not overlap.is_empty:
                    overlap_area = overlap.area
                    ortho_area = ortho_box.area
                    pct = 100 * overlap_area / shp_box.area if shp_box.area > 0 else 0
                    # Count features in overlap
                    features_in = gdf_reproj[gdf_reproj.intersects(ortho_box)]
                    print(f'    OVERLAPS {ortho["name"]}: {len(features_in)} features ({pct:.1f}% of shapefile)')
                else:
                    print(f'    No overlap with {ortho["name"]}')
        except Exception as e:
            print(f'  {name}: Reproject error: {e}')


# ============================================================
# PART 5: IDENTIFY WHICH VILLAGES EACH ORTHOPHOTO COVERS
# ============================================================
print(f'\n\n{"=" * 80}')
print("PART 5: WHICH VILLAGES DOES EACH ORTHOPHOTO COVER?")
print("=" * 80)

if buildings is not None and 'Village' in buildings.columns and ortho_info:
    buildings_reproj = buildings.to_crs(target_crs)
    
    for ortho in ortho_info:
        ob = ortho['bounds']
        ortho_box = box(ob.left, ob.bottom, ob.right, ob.top)
        
        buildings_in = buildings_reproj[buildings_reproj.intersects(ortho_box)]
        
        if len(buildings_in) > 0:
            villages_in = buildings_in['Village'].value_counts().to_dict()
            print(f'\n  {ortho["name"]}:')
            print(f'    Villages: {villages_in}')
            print(f'    Total buildings: {len(buildings_in)}')
        else:
            print(f'\n  {ortho["name"]}: No buildings found in this extent')

# Summary
print(f'\n\n{"=" * 80}')
print("SUMMARY")
print("=" * 80)
print(f'  Shapefiles: {len(all_gdfs)} loaded')
for name, gdf in all_gdfs.items():
    print(f'    {name}: {len(gdf)} features, {gdf.geom_type.iloc[0] if len(gdf) > 0 else "EMPTY"}')
print(f'  Orthophotos: {len(ortho_info)} found')
for o in ortho_info:
    print(f'    {o["name"]}: {o["width"]}x{o["height"]}')
