"""
Build script to create standalone executable for Satellite Classifier
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'pyinstaller',
        'opencv-python',
        'scikit-learn',
        'scikit-image',
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def clean_build_directories():
    """Clean previous build directories."""
    directories_to_clean = ['build', 'dist', '__pycache__']
    
    for directory in directories_to_clean:
        if os.path.exists(directory):
            print(f"Cleaning {directory}/")
            shutil.rmtree(directory)
    
    # Also clean .spec files
    for spec_file in Path('.').glob('*.spec'):
        print(f"Removing {spec_file}")
        spec_file.unlink()

def create_pyinstaller_spec():
    """Create PyInstaller spec file for the application."""
    spec_content = '''
# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path.cwd()))

block_cipher = None

# Define the analysis
a = Analysis(
    ['main.py'],
    pathex=[str(Path.cwd())],
    binaries=[],
    datas=[
        # Include any data files if needed
    ],
    hiddenimports=[
        # Core application modules
        'gui.main_window',
        'gui.progress_dialog',
        'gui.results_window',
        'preprocessing.image_processor',
        'preprocessing.noise_handler',
        'preprocessing.object_detector',
        'features.feature_extractor',
        'features.solar_panel_detector',
        'features.shape_analyzer',
        'features.texture_analyzer',
        'ml.classifier_ensemble',
        'ml.data_manager',
        'utils.logger',
        'utils.config',
        
        # Scientific libraries
        'sklearn',
        'sklearn.ensemble',
        'sklearn.tree',
        'sklearn.neighbors',
        'sklearn.svm',
        'sklearn.naive_bayes',
        'sklearn.linear_model',
        'sklearn.metrics',
        'sklearn.preprocessing',
        'sklearn.model_selection',
        'sklearn.cluster',
        'skimage',
        'skimage.feature',
        'skimage.measure',
        'scipy',
        'scipy.ndimage',
        'scipy.signal',
        'scipy.stats',
        'cv2',
        'numpy',
        'pandas',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'PIL',
        'joblib',
        
        # Tkinter components
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'IPython',
        'jupyter',
        'notebook',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'wx',
        'django',
        'flask',
        'tornado',
        'twisted',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out unnecessary files
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SatelliteClassifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon file path if available
)
'''
    
    with open('SatelliteClassifier.spec', 'w') as f:
        f.write(spec_content)
    
    print("Created PyInstaller spec file: SatelliteClassifier.spec")

def build_executable():
    """Build the executable using PyInstaller."""
    print("Building executable...")
    
    # Use the spec file for building
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--clean',
        '--noconfirm',
        'SatelliteClassifier.spec'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build completed successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print("Build failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def post_build_cleanup():
    """Perform post-build cleanup and organization."""
    # Check if build was successful
    exe_path = Path('dist/SatelliteClassifier.exe')
    
    if not exe_path.exists():
        print("Error: Executable not found in dist/")
        return False
    
    # Get file size
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(f"Executable size: {size_mb:.1f} MB")
    
    # Create release directory
    release_dir = Path('release')
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir()
    
    # Copy executable to release directory
    shutil.copy2(exe_path, release_dir / 'SatelliteClassifier.exe')
    
    # Create README for the release
    readme_content = """
# Satellite Object Classifier

## About
This is a standalone executable for classifying satellite objects using machine learning.

## Usage
1. Run SatelliteClassifier.exe
2. Select a directory containing three folders:
   - "Carrier Rockets" (with rocket images)
   - "Satellites" (with satellite images) 
   - "Debris" (with debris images)
3. Configure processing options as needed
4. Click "Start Classification" to begin
5. View results when processing completes

## Requirements
- Windows 10 or later
- At least 4GB RAM recommended
- Sufficient disk space for processing

## Input Format
- Supported image formats: JPG, PNG, BMP, TIFF
- Images should be of satellite objects
- Minimum 10 images per class recommended

## Output
- Classification results with accuracy metrics
- Feature analysis and confusion matrix
- Optional CSV export of extracted features

## Support
For issues or questions, please refer to the application logs in the "logs" directory.

Version: 1.0.0
Build Date: 2025-06-26
"""
    
    with open(release_dir / 'README.txt', 'w') as f:
        f.write(readme_content)
    
    print(f"Release package created in: {release_dir.absolute()}")
    print("Files included:")
    for file in release_dir.iterdir():
        print(f"  - {file.name}")
    
    return True

def main():
    """Main build process."""
    print("=" * 60)
    print("Satellite Classifier - Executable Build Script")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path('main.py').exists():
        print("Error: main.py not found. Run this script from the project root directory.")
        sys.exit(1)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ All dependencies are installed")
    
    # Clean previous builds
    print("\nCleaning previous builds...")
    clean_build_directories()
    print("✓ Clean completed")
    
    # Create spec file
    print("\nCreating PyInstaller specification...")
    create_pyinstaller_spec()
    print("✓ Spec file created")
    
    # Build executable
    print("\nBuilding executable (this may take several minutes)...")
    if not build_executable():
        print("✗ Build failed")
        sys.exit(1)
    print("✓ Build successful")
    
    # Post-build steps
    print("\nPerforming post-build cleanup...")
    if not post_build_cleanup():
        print("✗ Post-build cleanup failed")
        sys.exit(1)
    print("✓ Post-build cleanup completed")
    
    print("\n" + "=" * 60)
    print("BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("The standalone executable is ready in the 'release' directory.")
    print("You can distribute 'SatelliteClassifier.exe' as a single file.")
    print("\nTo test the executable:")
    print("1. Navigate to the release directory")
    print("2. Run SatelliteClassifier.exe")
    print("3. Select a test dataset directory")
    print("\nNote: First run may take longer as libraries are loaded.")

if __name__ == "__main__":
    main()
