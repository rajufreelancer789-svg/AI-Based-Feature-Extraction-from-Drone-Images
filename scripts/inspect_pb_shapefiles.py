"""
Deep inspection of Punjab (PB) shapefiles.
Compares schema with CG shapefiles to identify differences.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import numpy as np
from pathlib import Path

SHP_DIR = Path('/Users/appalaraju/IITT_AIML/pb-shp-file')
CG_SHP_DIR = Path('/Users/appalaraju/IITT_AIML/shp-file')

def inspect_shapefile(shp_path):
    """Inspect a single shapefile in detail."""
    name = shp_path.stem
    print(f'\n{"="*70}')
    print(f'  {name}')
    print(f'{"="*70}')
    
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        print(f'  ERROR: {e}')
        return None
    
    print(f'  Features: {len(gdf)}')
    print(f'  CRS: {gdf.crs}')
    
    if len(gdf) == 0:
        print(f'  ** EMPTY - no features **')
        return {'name': name, 'count': 0, 'crs': str(gdf.crs), 'columns': list(gdf.columns)}
    
    # Geometry types
    gtypes = gdf.geom_type.unique().tolist()
    print(f'  Geometry types: {gtypes}')
    
    # Columns
    cols = [c for c in gdf.columns if c != 'geometry']
    print(f'  Columns: {cols}')
    
    # Column details
    for col in cols:
        series = gdf[col]
        dtype = series.dtype
        nunique = series.nunique()
        nnulls = series.isna().sum()
        print(f'    {col}: dtype={dtype}, unique={nunique}, nulls={nnulls}')
        
        # Show value distribution for low-cardinality columns
        if nunique <= 20 and nunique > 0:
            if dtype in ('object', 'str'):
                vc = series.dropna().value_counts().to_dict()
            else:
                vc = series.value_counts().to_dict()
            print(f'      Values: {vc}')
    
    # Geometry validity
    invalid = (~gdf.geometry.is_valid).sum()
    empty = gdf.geometry.is_empty.sum()
    print(f'  Invalid geometries: {invalid}')
    print(f'  Empty geometries: {empty}')
    
    # Bounds
    total_bounds = gdf.total_bounds
    print(f'  Bounds ({gdf.crs}):')
    print(f'    X: [{total_bounds[0]:.2f}, {total_bounds[2]:.2f}]')
    print(f'    Y: [{total_bounds[1]:.2f}, {total_bounds[3]:.2f}]')
    
    # Area/Length stats
    first_type = str(gtypes[0]) if gtypes else ''
    if 'Polygon' in first_type or 'MultiPolygon' in first_type:
        areas = gdf.geometry.area
        print(f'  Area (sq units in {gdf.crs}):')
        print(f'    min={areas.min():.2f}, max={areas.max():.2f}')
        print(f'    mean={areas.mean():.2f}, median={areas.median():.2f}')
    elif 'LineString' in first_type or 'MultiLineString' in first_type:
        lengths = gdf.geometry.length
        print(f'  Length (units in {gdf.crs}):')
        print(f'    min={lengths.min():.2f}, max={lengths.max():.2f}')
        print(f'    mean={lengths.mean():.2f}, total={lengths.sum():.2f}')
    
    # Sample rows
    print(f'  Sample rows:')
    for idx in range(min(3, len(gdf))):
        row = gdf.iloc[idx]
        d = {c: row[c] for c in cols}
        d['_geom_type'] = row.geometry.geom_type if row.geometry is not None else 'NULL'
        print(f'    {d}')
    
    return {
        'name': name,
        'count': len(gdf),
        'crs': str(gdf.crs),
        'columns': cols,
        'geom_types': gtypes
    }


def compare_with_cg(pb_info_list, cg_shp_dir):
    """Compare PB and CG shapefile schemas."""
    print(f'\n{"="*70}')
    print(f'  COMPARISON: PB vs CG SCHEMAS')
    print(f'{"="*70}')
    
    # Load CG shapefile info
    cg_info = {}
    for shp_path in sorted(cg_shp_dir.glob('*.shp')):
        name = shp_path.stem
        try:
            gdf = gpd.read_file(shp_path)
            cg_info[name] = {
                'count': len(gdf),
                'columns': [c for c in gdf.columns if c != 'geometry'],
                'crs': str(gdf.crs)
            }
        except:
            pass
    
    pb_names = {info['name'] for info in pb_info_list if info}
    cg_names = set(cg_info.keys())
    
    print(f'\n  PB shapefiles: {sorted(pb_names)}')
    print(f'  CG shapefiles: {sorted(cg_names)}')
    print(f'\n  In PB but not CG: {sorted(pb_names - cg_names)}')
    print(f'  In CG but not PB: {sorted(cg_names - pb_names)}')
    
    # Compare columns for matching names
    for info in pb_info_list:
        if info is None:
            continue
        pb_name = info['name']
        
        # Try to find matching CG shapefile
        cg_match = None
        for cg_name in cg_names:
            if pb_name == cg_name:
                cg_match = cg_name
                break
            # Fuzzy match (e.g., Built_Up_Area_typ vs Built_Up_Area_type)
            if pb_name.startswith(cg_name[:10]) or cg_name.startswith(pb_name[:10]):
                cg_match = cg_name
                break
        
        if cg_match:
            pb_cols = set(info['columns'])
            cg_cols = set(cg_info[cg_match]['columns'])
            
            print(f'\n  {pb_name} (PB) vs {cg_match} (CG):')
            print(f'    PB features: {info["count"]}, CG features: {cg_info[cg_match]["count"]}')
            print(f'    PB CRS: {info["crs"]}, CG CRS: {cg_info[cg_match]["crs"]}')
            
            only_pb = pb_cols - cg_cols
            only_cg = cg_cols - pb_cols
            common = pb_cols & cg_cols
            
            if only_pb:
                print(f'    Columns ONLY in PB: {sorted(only_pb)}')
            if only_cg:
                print(f'    Columns ONLY in CG: {sorted(only_cg)}')
            if not only_pb and not only_cg:
                print(f'    Columns: IDENTICAL ({len(common)} columns)')
            else:
                print(f'    Common columns: {len(common)}')
        else:
            print(f'\n  {pb_name} (PB): NO MATCHING CG SHAPEFILE')


def main():
    print('='*80)
    print('  PUNJAB (PB) SHAPEFILE DEEP INSPECTION')
    print('='*80)
    
    # Part 1: Inspect each PB shapefile
    print(f'\n{"="*80}')
    print(f'PART 1: INDIVIDUAL SHAPEFILE DETAILS')
    print(f'{"="*80}')
    
    shp_files = sorted(SHP_DIR.glob('*.shp'))
    print(f'Found {len(shp_files)} shapefiles in {SHP_DIR}')
    
    pb_info_list = []
    for shp in shp_files:
        info = inspect_shapefile(shp)
        pb_info_list.append(info)
    
    # Part 2: Village analysis
    print(f'\n{"="*80}')
    print(f'PART 2: VILLAGE ANALYSIS (from Built_Up_Area files)')
    print(f'{"="*80}')
    
    for shp in shp_files:
        if 'Built_Up' in shp.stem:
            gdf = gpd.read_file(shp)
            # Check for village column
            village_cols = [c for c in gdf.columns if 'village' in c.lower() or 'gp_' in c.lower()]
            print(f'\n  {shp.stem} village-related columns: {village_cols}')
            for vc in village_cols:
                print(f'    {vc}: {gdf[vc].value_counts().to_dict()}')
    
    # Part 3: Compare with CG
    compare_with_cg(pb_info_list, CG_SHP_DIR)
    
    # Part 4: Summary
    print(f'\n{"="*80}')
    print(f'SUMMARY')
    print(f'{"="*80}')
    
    total_features = sum(info['count'] for info in pb_info_list if info)
    print(f'  Total PB features: {total_features}')
    for info in pb_info_list:
        if info:
            print(f'    {info["name"]}: {info["count"]} features')
    
    # Check naming issues that affect our pipeline
    print(f'\n  PIPELINE NAMING ISSUES:')
    expected_names = {'Built_Up_Area_type', 'Road', 'Road_Centre_Line', 'Water_Body', 
                      'Water_Body_Line', 'Waterbody_Point', 'Utility', 'Utility_Poly', 
                      'Bridge', 'Railway'}
    actual_names = {info['name'] for info in pb_info_list if info}
    
    mismatches = actual_names - expected_names
    missing = expected_names - actual_names
    if mismatches:
        print(f'    UNEXPECTED names (need mapping): {sorted(mismatches)}')
    if missing:
        print(f'    MISSING expected names: {sorted(missing)}')
    
    matched = actual_names & expected_names
    print(f'    Matching names: {sorted(matched)}')


if __name__ == '__main__':
    main()
