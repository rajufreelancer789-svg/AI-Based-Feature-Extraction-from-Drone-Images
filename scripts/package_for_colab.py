#!/usr/bin/env python3
"""
Package the project for upload to Google Drive / Colab.
Creates a zip with src/ and notebooks/ ready for Colab execution.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / 'colab_package.zip'

# Files to include
include = {
    'src/': ['.py'],
    'notebooks/': ['.ipynb'],
}

with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    for folder, extensions in include.items():
        folder_path = PROJECT_ROOT / folder
        for ext in extensions:
            for f in sorted(folder_path.glob(f'*{ext}')):
                arcname = f'{folder}{f.name}'
                zf.write(f, arcname)
                print(f'  + {arcname}')
    
    # Add __init__.py for src package
    init_path = PROJECT_ROOT / 'src' / '__init__.py'
    if init_path.exists():
        zf.write(init_path, 'src/__init__.py')
    else:
        zf.writestr('src/__init__.py', '')
        print('  + src/__init__.py (created)')

print(f'\nPackage created: {OUTPUT}')
print(f'Size: {OUTPUT.stat().st_size / 1024:.0f} KB')
print(f'''
UPLOAD INSTRUCTIONS:
1. Upload {OUTPUT.name} to Google Drive: MyDrive/IITT_AIML/
2. Upload orthophoto data to:
   - MyDrive/IITT_AIML/imagery/CG_train/  (CG orthophotos)
   - MyDrive/IITT_AIML/imagery/PB_train/  (PB orthophotos)
3. Upload shapefiles to:
   - MyDrive/IITT_AIML/shp-file/           (CG shapefiles)
   - MyDrive/IITT_AIML/imagery/PB_train/shp-file/  (PB shapefiles)
4. Open notebooks on Colab and run sequentially:
   01_data_setup → 02_preprocess → 03_train → 04_inference

OR simply upload the entire IITT_AIML/ folder to Google Drive.
''')
