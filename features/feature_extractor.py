"""
Comprehensive feature extraction for satellite object classification
"""

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import regionprops, label
from scipy import ndimage
from features.solar_panel_detector import SolarPanelDetector
from features.shape_analyzer import ShapeAnalyzer
from features.texture_analyzer import TextureAnalyzer
from utils.logger import get_logger

class FeatureExtractor:
    """Extracts comprehensive features from satellite images."""
    
    def __init__(self):
        self.logger = get_logger()
        self.solar_detector = SolarPanelDetector()
        self.shape_analyzer = ShapeAnalyzer()
        self.texture_analyzer = TextureAnalyzer()
        
        # Feature names for consistent ordering
        self.feature_names = [
            'solar_panel_area', 'solar_panel_count', 'edge_density', 'corner_count',
            'texture_contrast', 'texture_dissimilarity', 'texture_homogeneity', 'texture_energy',
            'aspect_ratio', 'circularity', 'solidity', 'extent',
            'object_area', 'convex_hull_area', 'perimeter', 'eccentricity',
            'hu_moment_1', 'hu_moment_2', 'hu_moment_3', 'hu_moment_4',
            'lbp_uniformity', 'euler_number', 'intensity_mean', 'intensity_std'
        ]
        
    def extract_all_features(self, image):
        """
        Extract all features from an image.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            
            # 1. Solar panel detection features
            solar_features = self.solar_detector.detect_solar_panels(image)
            features.update(solar_features)
            
            # 2. Shape analysis features
            shape_features = self.shape_analyzer.analyze_shape(image)
            features.update(shape_features)
            
            # 3. Texture analysis features
            texture_features = self.texture_analyzer.analyze_texture(image)
            features.update(texture_features)
            
            # 4. Edge and corner features
            edge_features = self._extract_edge_features(image)
            features.update(edge_features)
            
            # 5. Geometric features
            geometric_features = self._extract_geometric_features(image)
            features.update(geometric_features)
            
            # 6. Moment features
            moment_features = self._extract_moment_features(image)
            features.update(moment_features)
            
            # 7. Intensity statistics
            intensity_features = self._extract_intensity_features(image)
            features.update(intensity_features)
            
            # 8. Topological features
            topological_features = self._extract_topological_features(image)
            features.update(topological_features)
            
            # Ensure all features are present and in correct order
            ordered_features = {}
            for name in self.feature_names:
                ordered_features[name] = features.get(name, 0.0)
            
            return ordered_features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            # Return zero features in case of error
            return {name: 0.0 for name in self.feature_names}
            
    def _extract_edge_features(self, image):
        """Extract edge-related features."""
        # Canny edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Edge density
        edge_pixels = np.sum(edges > 0)
        total_pixels = image.shape[0] * image.shape[1]
        edge_density = edge_pixels / total_pixels
        
        # Corner detection using Harris corner detector
        corners = cv2.cornerHarris(image, 2, 3, 0.04)
        corner_count = np.sum(corners > 0.01 * corners.max())
        
        return {
            'edge_density': float(edge_density),
            'corner_count': float(corner_count)
        }
        
    def _extract_geometric_features(self, image):
        """Extract geometric features from the object."""
        # Threshold image to get binary mask
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'object_area': 0.0,
                'convex_hull_area': 0.0,
                'perimeter': 0.0,
                'aspect_ratio': 1.0,
                'extent': 0.0,
                'solidity': 0.0,
                'eccentricity': 0.0
            }
        
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Basic measurements
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Extent (ratio of object area to bounding rectangle area)
        extent = area / (w * h) if (w * h) > 0 else 0.0
        
        # Convex hull
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        
        # Solidity (ratio of object area to convex hull area)
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        # Fit ellipse and calculate eccentricity
        eccentricity = 0.0
        if len(largest_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(largest_contour)
                a, b = ellipse[1][0] / 2, ellipse[1][1] / 2  # Semi-major and semi-minor axes
                if a > 0 and b > 0:
                    eccentricity = np.sqrt(1 - (min(a, b) / max(a, b))**2)
            except:
                eccentricity = 0.0
        
        return {
            'object_area': float(area),
            'convex_hull_area': float(hull_area),
            'perimeter': float(perimeter),
            'aspect_ratio': float(aspect_ratio),
            'extent': float(extent),
            'solidity': float(solidity),
            'eccentricity': float(eccentricity)
        }
        
    def _extract_moment_features(self, image):
        """Extract moment-based features."""
        # Calculate Hu moments
        moments = cv2.moments(image)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log transform to make them more manageable
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return {
            'hu_moment_1': float(hu_moments[0]),
            'hu_moment_2': float(hu_moments[1]),
            'hu_moment_3': float(hu_moments[2]),
            'hu_moment_4': float(hu_moments[3])
        }
        
    def _extract_intensity_features(self, image):
        """Extract intensity-based statistical features."""
        # Basic statistics
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        return {
            'intensity_mean': float(mean_intensity),
            'intensity_std': float(std_intensity)
        }
        
    def _extract_topological_features(self, image):
        """Extract topological features."""
        # Threshold image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = binary // 255  # Convert to 0-1
        
        # Label connected components
        labeled_image = label(binary)
        
        # Calculate Euler number (connectivity measure)
        props = regionprops(labeled_image)
        euler_number = 0
        
        for prop in props:
            # Approximate Euler number using region properties
            euler_number += 1  # Each connected component contributes +1
        
        # LBP uniformity (texture regularity measure)
        lbp = local_binary_pattern(image, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
        lbp_uniformity = np.max(lbp_hist) / np.sum(lbp_hist) if np.sum(lbp_hist) > 0 else 0
        
        return {
            'euler_number': float(euler_number),
            'lbp_uniformity': float(lbp_uniformity)
        }
        
    def export_to_csv(self, features_data, output_path):
        """
        Export extracted features to CSV file.
        
        Args:
            features_data: List of (features_dict, label, image_path) tuples
            output_path: Path to save CSV file
        """
        try:
            rows = []
            
            for features, label, image_path in features_data:
                row = features.copy()
                row['label'] = label
                row['image_path'] = image_path
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Reorder columns to have label and path at the end
            feature_cols = [col for col in df.columns if col not in ['label', 'image_path']]
            df = df[feature_cols + ['label', 'image_path']]
            
            df.to_csv(output_path, index=False)
            self.logger.info(f"Features exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting features to CSV: {str(e)}")
            raise
