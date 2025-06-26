#!/usr/bin/env python3
"""
Console Demo for Satellite Object Classifier
Demonstrates the core functionality without GUI
"""

import os
import sys
import numpy as np
from datetime import datetime
import tempfile
import shutil

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.image_processor import ImageProcessor
from features.feature_extractor import FeatureExtractor
from ml.classifier_ensemble import ClassifierEnsemble
from ml.data_manager import DataManager
from utils.logger import setup_logger

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    print("Creating sample dataset for demonstration...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="satellite_demo_")
    
    # Create class directories
    classes = ['Carrier Rockets', 'Satellites', 'Debris']
    for class_name in classes:
        class_dir = os.path.join(temp_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create synthetic images (just noise for demo)
        for i in range(15):  # 15 images per class
            # Create a synthetic image with different characteristics per class
            if class_name == 'Carrier Rockets':
                # Elongated shape
                image = np.random.randint(0, 255, (200, 80), dtype=np.uint8)
                # Add some vertical structure
                image[80:120, 30:50] = 200
            elif class_name == 'Satellites':
                # More square with bright spots (solar panels)
                image = np.random.randint(0, 255, (120, 120), dtype=np.uint8)
                # Add bright rectangular regions
                image[40:60, 20:40] = 240
                image[40:60, 80:100] = 240
            else:  # Debris
                # Irregular shape
                image = np.random.randint(0, 255, (150, 90), dtype=np.uint8)
                # Add some random bright spots
                for _ in range(10):
                    x, y = np.random.randint(0, 150), np.random.randint(0, 90)
                    image[max(0, x-5):min(150, x+5), max(0, y-5):min(90, y+5)] = 255
            
            # Save as temporary file
            import cv2
            filename = f"sample_{i:02d}.png"
            filepath = os.path.join(class_dir, filename)
            cv2.imwrite(filepath, image)
    
    print(f"Sample dataset created at: {temp_dir}")
    return temp_dir

def demonstrate_processing():
    """Demonstrate the complete processing pipeline."""
    logger = setup_logger()
    print("=" * 80)
    print("SATELLITE OBJECT CLASSIFIER - CONSOLE DEMONSTRATION")
    print("=" * 80)
    print()
    
    dataset_path = None
    try:
        # Create sample dataset
        dataset_path = create_sample_dataset()
        
        print("1. INITIALIZING COMPONENTS")
        print("-" * 40)
        data_manager = DataManager()
        image_processor = ImageProcessor()
        feature_extractor = FeatureExtractor()
        classifier_ensemble = ClassifierEnsemble()
        print("✓ All components initialized successfully")
        print()
        
        print("2. LOADING AND SPLITTING DATASET")
        print("-" * 40)
        train_data, val_data, test_data = data_manager.load_and_split_data(
            dataset_path, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
        
        print(f"✓ Dataset loaded:")
        print(f"  - Training samples: {len(train_data)}")
        print(f"  - Validation samples: {len(val_data)}")
        print(f"  - Test samples: {len(test_data)}")
        
        # Show class distribution
        from collections import Counter
        train_dist = Counter([label for _, label in train_data])
        print(f"  - Training distribution: {dict(train_dist)}")
        print()
        
        print("3. IMAGE PREPROCESSING")
        print("-" * 40)
        processed_count = 0
        all_data = train_data + val_data + test_data
        processed_images = []
        
        for i, (image_path, label) in enumerate(all_data[:10]):  # Process first 10 for demo
            try:
                processed_img = image_processor.process_image(
                    image_path, apply_noise=True, remove_background=True)
                processed_images.append((processed_img, label, image_path))
                processed_count += 1
            except Exception as e:
                print(f"  - Error processing {image_path}: {str(e)}")
        
        print(f"✓ Processed {processed_count} images successfully")
        print("  - Applied environmental noise simulation")
        print("  - Performed denoising and background removal")
        print("  - Cropped to bounding boxes")
        print()
        
        print("4. FEATURE EXTRACTION")
        print("-" * 40)
        features_data = []
        feature_count = 0
        
        for i, (image, label, path) in enumerate(processed_images):
            try:
                features = feature_extractor.extract_all_features(image)
                features_data.append((features, label, path))
                feature_count += 1
            except Exception as e:
                print(f"  - Error extracting features from {path}: {str(e)}")
        
        print(f"✓ Extracted features from {feature_count} images")
        print(f"  - {len(feature_extractor.feature_names)} features per image:")
        
        # Show feature categories
        feature_categories = {
            'Solar Panel': ['solar_panel_area', 'solar_panel_count'],
            'Edge/Corner': ['edge_density', 'corner_count'],
            'Texture': ['texture_contrast', 'texture_dissimilarity', 'texture_homogeneity', 'texture_energy'],
            'Shape': ['aspect_ratio', 'circularity', 'solidity', 'extent', 'eccentricity'],
            'Geometric': ['object_area', 'convex_hull_area', 'perimeter'],
            'Moments': ['hu_moment_1', 'hu_moment_2', 'hu_moment_3', 'hu_moment_4'],
            'Intensity': ['intensity_mean', 'intensity_std'],
            'Topological': ['lbp_uniformity', 'euler_number']
        }
        
        for category, features in feature_categories.items():
            print(f"    * {category}: {len(features)} features")
        print()
        
        # Show sample features for one image
        if features_data:
            sample_features = features_data[0][0]
            print("  Sample feature values (first image):")
            for name, value in list(sample_features.items())[:8]:  # Show first 8
                print(f"    - {name}: {value:.4f}")
            print("    - ... (and 16 more features)")
        print()
        
        print("5. MACHINE LEARNING TRAINING")
        print("-" * 40)
        
        if len(features_data) < 6:
            print("⚠ Not enough data for proper ML training (need more samples)")
            print("  In real usage, ensure at least 50+ images per class")
            
            # Create minimal synthetic data for demonstration
            print("  Creating synthetic data for ML demo...")
            synthetic_features = []
            labels = ['Carrier Rockets', 'Satellites', 'Debris']
            
            for label in labels:
                for i in range(10):  # 10 samples per class
                    # Generate synthetic features based on class
                    features = {}
                    for feature_name in feature_extractor.feature_names:
                        if label == 'Carrier Rockets':
                            # High aspect ratio, low circularity
                            if feature_name == 'aspect_ratio':
                                features[feature_name] = np.random.normal(3.0, 0.5)
                            elif feature_name == 'circularity':
                                features[feature_name] = np.random.normal(0.3, 0.1)
                            else:
                                features[feature_name] = np.random.normal(0.5, 0.2)
                        elif label == 'Satellites':
                            # High solar panel area, medium circularity
                            if feature_name == 'solar_panel_area':
                                features[feature_name] = np.random.normal(0.2, 0.05)
                            elif feature_name == 'circularity':
                                features[feature_name] = np.random.normal(0.6, 0.1)
                            else:
                                features[feature_name] = np.random.normal(0.5, 0.2)
                        else:  # Debris
                            # Low solidity, high texture contrast
                            if feature_name == 'solidity':
                                features[feature_name] = np.random.normal(0.3, 0.1)
                            elif feature_name == 'texture_contrast':
                                features[feature_name] = np.random.normal(0.8, 0.1)
                            else:
                                features[feature_name] = np.random.normal(0.5, 0.2)
                        
                        # Ensure positive values
                        features[feature_name] = max(0, features[feature_name])
                    
                    synthetic_features.append((features, label, f"synthetic_{label}_{i}"))
            
            # Split synthetic data
            train_features = synthetic_features[:21]  # 7 per class
            val_features = synthetic_features[21:24]   # 1 per class
            test_features = synthetic_features[24:]    # 2 per class
            
        else:
            # Use real extracted features
            train_features = features_data[:len(train_data)]
            val_features = features_data[len(train_data):len(train_data)+len(val_data)]
            test_features = features_data[len(train_data)+len(val_data):]
        
        print("✓ Prepared training data")
        print(f"  - Training features: {len(train_features)}")
        print(f"  - Validation features: {len(val_features)}")
        print(f"  - Test features: {len(test_features)}")
        print()
        
        print("  Training ensemble of 7 classifiers:")
        classifiers = [
            "Random Forest", "Decision Tree", "K-Nearest Neighbors",
            "Support Vector Machine", "Gradient Boosting", 
            "Naive Bayes", "Logistic Regression"
        ]
        for clf in classifiers:
            print(f"    * {clf}")
        print()
        
        # Train the ensemble
        print("  Training in progress...")
        results = classifier_ensemble.train_and_evaluate(
            train_features, val_features, test_features)
        
        print("✓ Training completed successfully!")
        print()
        
        print("6. CLASSIFICATION RESULTS")
        print("-" * 40)
        print(f"✓ Ensemble Accuracy: {results['ensemble_accuracy']:.4f} ({results['ensemble_accuracy']*100:.2f}%)")
        print(f"✓ Ensemble Precision: {results['ensemble_precision']:.4f}")
        print(f"✓ Ensemble Recall: {results['ensemble_recall']:.4f}")
        print(f"✓ Ensemble F1-Score: {results['ensemble_f1']:.4f}")
        print()
        
        print("  Individual Classifier Performance:")
        for name, metrics in results['individual_results'].items():
            weight = results['weights'].get(name, 0.0)
            print(f"    * {name}:")
            print(f"      - Accuracy: {metrics['accuracy']:.4f} | Weight: {weight:.4f}")
        print()
        
        print("  Confusion Matrix (Ensemble):")
        cm = results['confusion_matrix']
        classes = ['Carrier Rockets', 'Satellites', 'Debris']
        print("    Predicted →")
        print("    Actual ↓     ", end="")
        for cls in classes:
            print(f"{cls[:8]:>8}", end=" ")
        print()
        
        for i, actual_cls in enumerate(classes):
            print(f"    {actual_cls[:12]:12} ", end="")
            for j in range(len(classes)):
                print(f"{cm[i][j]:8d}", end=" ")
            print()
        print()
        
        # Feature importance (if available)
        if 'feature_importance' in results:
            importance = results['feature_importance']
            top_features = sorted(zip(feature_extractor.feature_names, importance), 
                                key=lambda x: x[1], reverse=True)
            
            print("  Top 10 Most Important Features:")
            for i, (feature_name, importance_val) in enumerate(top_features[:10]):
                print(f"    {i+1:2d}. {feature_name:20} ({importance_val:.4f})")
        print()
        
        print("7. FEATURE EXPORT")
        print("-" * 40)
        csv_path = "demo_features.csv"
        feature_extractor.export_to_csv(train_features + val_features + test_features, csv_path)
        print(f"✓ Features exported to: {csv_path}")
        print(f"  - {len(train_features + val_features + test_features)} samples")
        print(f"  - {len(feature_extractor.feature_names)} features per sample")
        print()
        
        print("8. PREDICTION EXAMPLE")
        print("-" * 40)
        if test_features:
            sample_features = test_features[0][0]
            true_label = test_features[0][1]
            predicted_label = classifier_ensemble.predict(sample_features)
            
            print(f"✓ Sample prediction:")
            print(f"  - True label: {true_label}")
            print(f"  - Predicted label: {predicted_label}")
            print(f"  - Prediction correct: {'Yes' if true_label == predicted_label else 'No'}")
        print()
        
    except Exception as e:
        print(f"✗ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            if 'dataset_path' in locals():
                shutil.rmtree(dataset_path)
                print("✓ Cleaned up temporary files")
        except:
            pass
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETED")
    print("=" * 80)
    print()
    print("This demonstrates the core functionality of the Satellite Classifier:")
    print("• Complete image preprocessing pipeline with noise handling")
    print("• Advanced feature extraction (24 features covering multiple domains)")
    print("• Ensemble machine learning with 7 different algorithms")
    print("• Weighted voting system based on individual classifier performance")
    print("• Comprehensive results analysis and visualization")
    print()
    print("For the full GUI application, run: python main.py")
    print("For standalone executable creation, run: python build_exe.py")

if __name__ == "__main__":
    demonstrate_processing()