"""
Shape analysis for satellite object classification
"""

import cv2
import numpy as np
from skimage.measure import regionprops, label
from utils.logger import get_logger

class ShapeAnalyzer:
    """Analyzes shape characteristics of satellite objects."""
    
    def __init__(self):
        self.logger = get_logger()
        
    def analyze_shape(self, image):
        """
        Analyze shape characteristics of the object in the image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Dictionary with shape features
        """
        try:
            # Get binary mask of the object
            binary_mask = self._get_object_mask(image)
            
            if np.sum(binary_mask) == 0:
                return self._empty_shape_features()
            
            # Find main contour
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return self._empty_shape_features()
            
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate shape features
            features = {}
            
            # Basic geometric features
            features.update(self._calculate_basic_geometry(main_contour))
            
            # Circularity and compactness
            features.update(self._calculate_circularity(main_contour))
            
            # Elongation and orientation features
            features.update(self._calculate_elongation_features(main_contour))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in shape analysis: {str(e)}")
            return self._empty_shape_features()
            
    def _get_object_mask(self, image):
        """Extract binary mask of the main object."""
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Also try Otsu thresholding
        _, otsu_binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine both methods
        combined = cv2.bitwise_or(binary, otsu_binary)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        return combined
        
    def _calculate_basic_geometry(self, contour):
        """Calculate basic geometric properties."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Extent (ratio of object area to bounding rectangle area)
        extent = area / (w * h) if (w * h) > 0 else 0.0
        
        return {
            'aspect_ratio': float(aspect_ratio),
            'extent': float(extent)
        }
        
    def _calculate_circularity(self, contour):
        """Calculate circularity and related shape measures."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity (4π * area / perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Solidity (ratio of object area to convex hull area)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'circularity': float(circularity),
            'solidity': float(solidity)
        }
        
    def _calculate_elongation_features(self, contour):
        """Calculate elongation and orientation features."""
        if len(contour) < 5:
            return {
                'eccentricity': 0.0,
                'orientation': 0.0,
                'major_axis_length': 0.0,
                'minor_axis_length': 0.0
            }
        
        try:
            # Fit ellipse to contour
            ellipse = cv2.fitEllipse(contour)
            
            # Extract ellipse parameters
            center, axes, angle = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            
            # Eccentricity
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis)**2)
            else:
                eccentricity = 0
            
            # Orientation (angle of major axis)
            orientation = angle
            
            return {
                'eccentricity': float(eccentricity),
                'orientation': float(orientation),
                'major_axis_length': float(major_axis),
                'minor_axis_length': float(minor_axis)
            }
            
        except:
            # Fallback if ellipse fitting fails
            return {
                'eccentricity': 0.0,
                'orientation': 0.0,
                'major_axis_length': 0.0,
                'minor_axis_length': 0.0
            }
            
    def _empty_shape_features(self):
        """Return empty shape features for error cases."""
        return {
            'aspect_ratio': 1.0,
            'extent': 0.0,
            'circularity': 0.0,
            'solidity': 0.0,
            'eccentricity': 0.0,
            'orientation': 0.0,
            'major_axis_length': 0.0,
            'minor_axis_length': 0.0
        }
        
    def classify_shape_type(self, features):
        """
        Classify the general shape type based on features.
        
        Returns:
            String describing the shape type
        """
        circularity = features.get('circularity', 0)
        aspect_ratio = features.get('aspect_ratio', 1)
        solidity = features.get('solidity', 0)
        eccentricity = features.get('eccentricity', 0)
        
        # Circular objects (likely satellites with circular components)
        if circularity > 0.7 and aspect_ratio < 1.5:
            return "circular"
        
        # Elongated objects (likely rocket bodies)
        elif aspect_ratio > 3.0 or eccentricity > 0.8:
            return "elongated"
        
        # Rectangular/angular objects (likely satellites with solar panels)
        elif solidity > 0.8 and 1.5 < aspect_ratio < 3.0:
            return "rectangular"
        
        # Irregular objects (likely debris)
        elif solidity < 0.6 or circularity < 0.3:
            return "irregular"
        
        # Default
        else:
            return "complex"
