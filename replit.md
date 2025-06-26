# Satellite Object Classifier

## Overview

This is a Python-based machine learning application for classifying satellite objects into three categories: Carrier Rockets, Satellites, and Debris. The application uses computer vision techniques and ensemble machine learning to analyze satellite images and provide accurate classifications through a user-friendly GUI interface.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Tkinter-based GUI with multiple windows for user interaction
- **Backend**: Python-based processing pipeline with machine learning models
- **Data Processing**: Image preprocessing, feature extraction, and noise handling
- **Machine Learning**: Ensemble classifier with multiple algorithms
- **Configuration**: Centralized configuration management

## Key Components

### GUI Layer (`gui/`)
- **main_window.py**: Primary application interface with dataset selection and processing controls
- **progress_dialog.py**: Modal dialog for showing processing progress with status updates
- **results_window.py**: Tabbed interface displaying classification metrics and visualizations

The GUI provides an intuitive workflow where users select datasets, configure processing parameters, and view detailed results.

### Image Processing Pipeline (`preprocessing/`)
- **image_processor.py**: Main processing pipeline handling image standardization and enhancement
- **noise_handler.py**: Specialized noise simulation and removal for space environment conditions
- **object_detector.py**: Multi-strategy object detection using edge detection, clustering, and contour analysis

The preprocessing pipeline simulates realistic space conditions including cosmic ray noise, thermal sensor noise, and atmospheric scattering effects.

### Feature Extraction (`features/`)
- **feature_extractor.py**: Orchestrates comprehensive feature extraction across 24 different measurements
- **solar_panel_detector.py**: Specialized detection of solar panels using shape, reflectivity, and texture analysis
- **shape_analyzer.py**: Geometric shape analysis including circularity, aspect ratio, and Hu moments
- **texture_analyzer.py**: Texture analysis using GLCM, LBP, and Gabor filters

The feature extraction produces 24 distinct features covering geometric, textural, and domain-specific characteristics.

### Machine Learning (`ml/`)
- **classifier_ensemble.py**: Ensemble of 7 different classifiers with weighted voting
- **data_manager.py**: Dataset loading, validation, and train/validation/test splitting

The ensemble includes Random Forest, SVM, KNN, Gradient Boosting, Decision Trees, Naive Bayes, and Logistic Regression with cross-validation for robust predictions.

### Utilities (`utils/`)
- **config.py**: Centralized configuration for processing parameters and application settings
- **logger.py**: Logging infrastructure with file and console output

## Data Flow

1. **Dataset Selection**: User selects directory containing class-organized image folders
2. **Data Validation**: System validates dataset structure and image formats
3. **Data Splitting**: Images split into training (75%), validation (5%), and test (20%) sets
4. **Image Processing**: Each image undergoes noise simulation, denoising, and background removal
5. **Feature Extraction**: 24 features extracted per image covering shape, texture, and domain-specific characteristics
6. **Model Training**: Ensemble of 7 classifiers trained with cross-validation
7. **Evaluation**: Performance metrics calculated and visualized
8. **Results Display**: Comprehensive results shown in tabbed interface

## External Dependencies

- **OpenCV**: Image processing and computer vision operations
- **Scikit-learn**: Machine learning algorithms and evaluation metrics
- **Scikit-image**: Advanced image processing and feature extraction
- **NumPy/Pandas**: Numerical computing and data manipulation
- **Matplotlib**: Visualization and plotting for results display
- **SciPy**: Scientific computing functions
- **Joblib**: Model serialization and parallel processing

## Deployment Strategy

The application is designed for standalone deployment:

- **Development**: Replit environment with Nix package management
- **Distribution**: PyInstaller executable build script (`build_exe.py`)
- **Dependencies**: Self-contained with all required libraries bundled
- **Cross-platform**: Supports Windows, macOS, and Linux

The build script creates a single executable file with all dependencies included for easy distribution.

## Recent Changes

- **June 26, 2025**: Complete satellite classifier implementation finished
  - Full GUI application with tkinter interface
  - Advanced image preprocessing with noise simulation and denoising
  - Comprehensive 24-feature extraction system
  - Ensemble ML with 7 algorithms and weighted voting
  - Results visualization with confusion matrix and metrics
  - Standalone executable builder script
  - Console demonstration and architecture overview

## Changelog

- June 26, 2025. Complete satellite object classifier application implemented
- June 26, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.
Requirements: Standalone .exe application for satellite classification.