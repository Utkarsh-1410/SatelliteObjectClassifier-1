"""
Data management for satellite classification
"""

import os
import random
import glob
from collections import defaultdict
from utils.logger import get_logger

class DataManager:
    """Manages dataset loading, splitting, and organization."""
    
    def __init__(self):
        self.logger = get_logger()
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
    def load_and_split_data(self, dataset_path, train_ratio=0.75, val_ratio=0.05, test_ratio=0.20):
        """
        Load images from dataset directory and split into train/val/test sets.
        
        Args:
            dataset_path: Path to dataset directory containing class folders
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set  
            test_ratio: Ratio for test set
            
        Returns:
            Tuple of (train_data, val_data, test_data) where each is a list of (image_path, label) tuples
        """
        try:
            # Validate ratios
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError("Split ratios must sum to 1.0")
            
            # Load all images by class
            class_data = self._load_images_by_class(dataset_path)
            
            if not class_data:
                raise ValueError("No images found in dataset")
            
            # Split each class separately to maintain class balance
            train_data = []
            val_data = []
            test_data = []
            
            for class_name, image_paths in class_data.items():
                if not image_paths:
                    self.logger.warning(f"No images found for class: {class_name}")
                    continue
                
                # Shuffle images for random splitting
                shuffled_paths = image_paths.copy()
                random.shuffle(shuffled_paths)
                
                n_images = len(shuffled_paths)
                n_train = int(n_images * train_ratio)
                n_val = int(n_images * val_ratio)
                
                # Split the data
                train_paths = shuffled_paths[:n_train]
                val_paths = shuffled_paths[n_train:n_train + n_val]
                test_paths = shuffled_paths[n_train + n_val:]
                
                # Add to respective sets with labels
                train_data.extend([(path, class_name) for path in train_paths])
                val_data.extend([(path, class_name) for path in val_paths])
                test_data.extend([(path, class_name) for path in test_paths])
                
                self.logger.info(f"Class '{class_name}': {len(train_paths)} train, "
                               f"{len(val_paths)} val, {len(test_paths)} test")
            
            # Shuffle the final datasets
            random.shuffle(train_data)
            random.shuffle(val_data)
            random.shuffle(test_data)
            
            self.logger.info(f"Dataset split complete - Total: Train={len(train_data)}, "
                           f"Val={len(val_data)}, Test={len(test_data)}")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            self.logger.error(f"Error loading and splitting data: {str(e)}")
            raise
            
    def _load_images_by_class(self, dataset_path):
        """Load all image paths organized by class."""
        class_data = defaultdict(list)
        
        # Expected class folders
        class_folders = ['Carrier Rockets', 'Satellites', 'Debris']
        
        for class_name in class_folders:
            class_path = os.path.join(dataset_path, class_name)
            
            if not os.path.exists(class_path):
                self.logger.warning(f"Class folder not found: {class_path}")
                continue
            
            # Find all image files in the class folder
            image_paths = []
            
            for ext in self.supported_extensions:
                # Search for images with current extension
                pattern = os.path.join(class_path, f"*{ext}")
                found_files = glob.glob(pattern)
                image_paths.extend(found_files)
                
                # Also search case-insensitive
                pattern = os.path.join(class_path, f"*{ext.upper()}")
                found_files = glob.glob(pattern)
                image_paths.extend(found_files)
            
            # Remove duplicates and sort
            image_paths = sorted(list(set(image_paths)))
            
            self.logger.info(f"Found {len(image_paths)} images for class '{class_name}'")
            class_data[class_name] = image_paths
        
        return dict(class_data)
        
    def get_class_distribution(self, data_list):
        """Get the distribution of classes in a dataset."""
        class_counts = defaultdict(int)
        
        for _, label in data_list:
            class_counts[label] += 1
        
        return dict(class_counts)
        
    def validate_dataset(self, dataset_path):
        """
        Validate that the dataset has the correct structure and sufficient data.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'class_counts': {},
            'total_images': 0
        }
        
        try:
            # Check if dataset path exists
            if not os.path.exists(dataset_path):
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Dataset path does not exist: {dataset_path}")
                return validation_results
            
            # Expected class folders
            expected_classes = ['Carrier Rockets', 'Satellites', 'Debris']
            
            for class_name in expected_classes:
                class_path = os.path.join(dataset_path, class_name)
                
                if not os.path.exists(class_path):
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"Missing class folder: {class_name}")
                    continue
                
                # Count images in class folder
                image_count = 0
                for ext in self.supported_extensions:
                    pattern = os.path.join(class_path, f"*{ext}")
                    image_count += len(glob.glob(pattern))
                    pattern = os.path.join(class_path, f"*{ext.upper()}")
                    image_count += len(glob.glob(pattern))
                
                validation_results['class_counts'][class_name] = image_count
                validation_results['total_images'] += image_count
                
                # Check for minimum number of images
                if image_count < 10:
                    validation_results['warnings'].append(
                        f"Class '{class_name}' has only {image_count} images. "
                        "Consider adding more images for better training."
                    )
                
                if image_count == 0:
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"No images found in class '{class_name}'")
            
            # Check for class imbalance
            if validation_results['class_counts']:
                counts = list(validation_results['class_counts'].values())
                max_count = max(counts)
                min_count = min(counts)
                
                if max_count > 0 and min_count / max_count < 0.5:
                    validation_results['warnings'].append(
                        "Dataset is imbalanced. Consider balancing the number of images per class."
                    )
            
            # Check total dataset size
            if validation_results['total_images'] < 50:
                validation_results['warnings'].append(
                    f"Dataset is small ({validation_results['total_images']} images). "
                    "Consider adding more images for better model performance."
                )
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Error during validation: {str(e)}")
        
        return validation_results
        
    def augment_dataset(self, data_list, augmentation_factor=2):
        """
        Simple dataset augmentation by duplicating data.
        In a real implementation, this would apply image transformations.
        
        Args:
            data_list: List of (image_path, label) tuples
            augmentation_factor: How many times to replicate the data
            
        Returns:
            Augmented data list
        """
        augmented_data = data_list.copy()
        
        for _ in range(augmentation_factor - 1):
            augmented_data.extend(data_list)
        
        # Shuffle augmented data
        random.shuffle(augmented_data)
        
        self.logger.info(f"Dataset augmented from {len(data_list)} to {len(augmented_data)} samples")
        
        return augmented_data
