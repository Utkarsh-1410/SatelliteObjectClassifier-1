"""
Configuration settings for Satellite Classifier Application
"""

import os

# Application Information
APP_NAME = "Satellite Object Classifier"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-powered satellite object classification system"

# Default Processing Parameters
DEFAULT_TRAIN_SPLIT = 75.0
DEFAULT_VAL_SPLIT = 5.0
DEFAULT_TEST_SPLIT = 20.0

# Image Processing Settings
DEFAULT_IMAGE_SIZE = (256, 256)
PROCESSED_IMAGE_SIZE = (128, 128)
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

# Feature Extraction Parameters
N_FEATURES = 24
FEATURE_NAMES = [
    'solar_panel_area', 'solar_panel_count', 'edge_density', 'corner_count',
    'texture_contrast', 'texture_dissimilarity', 'texture_homogeneity', 'texture_energy',
    'aspect_ratio', 'circularity', 'solidity', 'extent',
    'object_area', 'convex_hull_area', 'perimeter', 'eccentricity',
    'hu_moment_1', 'hu_moment_2', 'hu_moment_3', 'hu_moment_4',
    'lbp_uniformity', 'euler_number', 'intensity_mean', 'intensity_std'
]

# Class Information
CLASS_NAMES = ['Carrier Rockets', 'Satellites', 'Debris']
N_CLASSES = len(CLASS_NAMES)

# Noise Processing Parameters
GAUSSIAN_NOISE_STD = 5
COSMIC_RAY_PROBABILITY = 0.001
THERMAL_NOISE_STD = 3
ATMOSPHERIC_EFFECT_PROBABILITY = 0.3

# Edge Detection Parameters
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Texture Analysis Parameters
LBP_RADIUS = 3
LBP_N_POINTS = 24
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, 45, 90, 135]  # in degrees
GLCM_LEVELS = 8

# Solar Panel Detection Parameters
MIN_PANEL_AREA_RATIO = 0.001  # Minimum panel area as fraction of image
MAX_PANEL_AREA_RATIO = 0.1    # Maximum panel area as fraction of image
PANEL_ASPECT_RATIO_MIN = 1.2
PANEL_ASPECT_RATIO_MAX = 5.0
PANEL_UNIFORMITY_THRESHOLD = 0.7

# Shape Analysis Parameters
MIN_OBJECT_AREA_RATIO = 0.005  # Minimum object area as fraction of image
MAX_OBJECT_AREA_RATIO = 0.8    # Maximum object area as fraction of image
CIRCULARITY_THRESHOLD = 0.1    # Minimum circularity for valid contours

# Machine Learning Parameters
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = 10
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 5

DECISION_TREE_MAX_DEPTH = 8
DECISION_TREE_MIN_SAMPLES_SPLIT = 10

KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = 'distance'

SVM_C = 1.0
SVM_GAMMA = 'scale'
SVM_KERNEL = 'rbf'

GRADIENT_BOOSTING_N_ESTIMATORS = 100
GRADIENT_BOOSTING_LEARNING_RATE = 0.1
GRADIENT_BOOSTING_MAX_DEPTH = 6

LOGISTIC_REGRESSION_C = 1.0
LOGISTIC_REGRESSION_MAX_ITER = 1000

# GUI Settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MIN_WINDOW_WIDTH = 600
MIN_WINDOW_HEIGHT = 500

LOG_DISPLAY_HEIGHT = 15
PROGRESS_UPDATE_INTERVAL = 100  # milliseconds

# File Paths and Directories
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MODEL_DIR = "models"
DEFAULT_LOG_DIR = "logs"
DEFAULT_FEATURES_FILENAME = "extracted_features.csv"
DEFAULT_MODEL_FILENAME = "satellite_classifier_model.joblib"
DEFAULT_LOG_FILENAME = "satellite_classifier.log"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# Performance Settings
MAX_WORKERS = 4  # For parallel processing
MEMORY_LIMIT_MB = 1024  # Memory usage warning threshold
PROCESSING_TIMEOUT = 3600  # Maximum processing time in seconds (1 hour)

# Validation Settings
MIN_IMAGES_PER_CLASS = 10
MIN_TOTAL_IMAGES = 50
CLASS_IMBALANCE_THRESHOLD = 0.5  # Min ratio of smallest to largest class

# Error Messages
ERROR_MESSAGES = {
    'invalid_directory': "Please select a valid directory containing 'Carrier Rockets', 'Satellites', and 'Debris' folders.",
    'insufficient_data': "Insufficient data found. Each class should have at least {} images.",
    'processing_failed': "Processing failed due to an unexpected error.",
    'memory_error': "Insufficient memory to process the dataset. Try reducing the dataset size.",
    'timeout_error': "Processing timed out. The dataset may be too large.",
    'invalid_image': "Invalid or corrupted image file found.",
    'no_features': "Failed to extract features from images.",
    'training_failed': "Model training failed.",
    'prediction_failed': "Prediction failed.",
    'file_not_found': "Required file not found.",
    'permission_error': "Permission denied. Check file access permissions."
}

# Success Messages
SUCCESS_MESSAGES = {
    'processing_complete': "Satellite classification completed successfully!",
    'features_exported': "Features exported to CSV file successfully.",
    'model_saved': "Classification model saved successfully.",
    'dataset_loaded': "Dataset loaded and validated successfully."
}

# Help Text
HELP_TEXT = {
    'directory_selection': "Select a directory containing three folders named 'Carrier Rockets', 'Satellites', and 'Debris' with satellite images.",
    'data_splits': "Adjust the percentage split for training, validation, and testing data. Values should sum to 100%.",
    'noise_processing': "Enable to add realistic space environment noise and then remove it to improve classification robustness.",
    'background_removal': "Enable to automatically remove image backgrounds and focus on the satellite objects.",
    'feature_export': "Enable to save extracted features to a CSV file for analysis.",
    'ensemble_classification': "Uses multiple machine learning algorithms with weighted voting for robust classification."
}

# Development Settings (for debugging)
DEBUG_MODE = False
VERBOSE_LOGGING = False
SAVE_INTERMEDIATE_IMAGES = False
PROFILE_PERFORMANCE = False

# Ensure output directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [DEFAULT_OUTPUT_DIR, DEFAULT_MODEL_DIR, DEFAULT_LOG_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError:
                pass  # Directory might already exist or permission error

# Version and build information
BUILD_DATE = "2025-06-26"
PYTHON_VERSION_REQUIRED = "3.7"
DEPENDENCIES_VERSION = {
    'opencv-python': '>=4.5.0',
    'scikit-learn': '>=1.0.0',
    'scikit-image': '>=0.18.0',
    'numpy': '>=1.19.0',
    'pandas': '>=1.3.0',
    'matplotlib': '>=3.3.0',
    'scipy': '>=1.7.0'
}
