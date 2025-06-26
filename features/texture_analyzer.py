"""
Texture analysis for satellite object classification
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy import ndimage
from scipy.stats import entropy
from utils.logger import get_logger

class TextureAnalyzer:
    """Analyzes texture characteristics of satellite objects."""
    
    def __init__(self):
        self.logger = get_logger()
        
    def analyze_texture(self, image):
        """
        Analyze texture characteristics of the image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Dictionary with texture features
        """
        try:
            features = {}
            
            # Gray Level Co-occurrence Matrix (GLCM) features
            glcm_features = self._calculate_glcm_features(image)
            features.update(glcm_features)
            
            # Local Binary Pattern (LBP) features
            lbp_features = self._calculate_lbp_features(image)
            features.update(lbp_features)
            
            # Gabor filter responses
            gabor_features = self._calculate_gabor_features(image)
            features.update(gabor_features)
            
            # Statistical texture measures
            stats_features = self._calculate_statistical_features(image)
            features.update(stats_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in texture analysis: {str(e)}")
            return self._empty_texture_features()
            
    def _calculate_glcm_features(self, image):
        """Calculate Gray Level Co-occurrence Matrix features."""
        # Reduce image levels for GLCM calculation
        image_reduced = (image // 32).astype(np.uint8)  # Reduce to 8 levels
        
        # Calculate GLCM for different distances and angles
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        contrast_values = []
        dissimilarity_values = []
        homogeneity_values = []
        energy_values = []
        
        for distance in distances:
            for angle in angles:
                try:
                    glcm = graycomatrix(image_reduced, [distance], [angle], 
                                     levels=8, symmetric=True, normed=True)
                    
                    contrast_values.append(graycoprops(glcm, 'contrast')[0, 0])
                    dissimilarity_values.append(graycoprops(glcm, 'dissimilarity')[0, 0])
                    homogeneity_values.append(graycoprops(glcm, 'homogeneity')[0, 0])
                    energy_values.append(graycoprops(glcm, 'energy')[0, 0])
                    
                except Exception:
                    # Skip this distance/angle combination if it fails
                    continue
        
        # Use mean values across all distances and angles
        return {
            'texture_contrast': float(np.mean(contrast_values)) if contrast_values else 0.0,
            'texture_dissimilarity': float(np.mean(dissimilarity_values)) if dissimilarity_values else 0.0,
            'texture_homogeneity': float(np.mean(homogeneity_values)) if homogeneity_values else 0.0,
            'texture_energy': float(np.mean(energy_values)) if energy_values else 0.0
        }
        
    def _calculate_lbp_features(self, image):
        """Calculate Local Binary Pattern features."""
        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # Calculate histogram
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Normalize histogram
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        # LBP uniformity (measure of texture regularity)
        uniformity = np.max(hist)
        
        # LBP entropy (measure of texture complexity)
        lbp_entropy = entropy(hist + 1e-7)
        
        return {
            'lbp_uniformity': float(uniformity),
            'lbp_entropy': float(lbp_entropy)
        }
        
    def _calculate_gabor_features(self, image):
        """Calculate Gabor filter responses."""
        # Define Gabor filter parameters
        frequencies = [0.1, 0.3, 0.5]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        responses = []
        
        for freq in frequencies:
            for theta in orientations:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel((21, 21), 5, theta, 2*np.pi/freq, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                
                # Calculate response statistics
                mean_response = np.mean(filtered)
                std_response = np.std(filtered)
                
                responses.append(mean_response)
                responses.append(std_response)
        
        # Use first few responses as features
        return {
            'gabor_response_1': float(responses[0]) if len(responses) > 0 else 0.0,
            'gabor_response_2': float(responses[1]) if len(responses) > 1 else 0.0,
            'gabor_response_3': float(responses[2]) if len(responses) > 2 else 0.0,
            'gabor_response_4': float(responses[3]) if len(responses) > 3 else 0.0
        }
        
    def _calculate_statistical_features(self, image):
        """Calculate statistical texture measures."""
        # Convert to float for calculations
        img_float = image.astype(np.float64)
        
        # Calculate gradients
        grad_x = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Edge density
        edge_density = np.mean(gradient_magnitude) / 255.0
        
        # Variance (measure of texture coarseness)
        variance = np.var(img_float) / (255.0**2)
        
        # Skewness (measure of asymmetry)
        mean_val = np.mean(img_float)
        skewness = np.mean(((img_float - mean_val) / 255.0)**3)
        
        # Kurtosis (measure of peakedness)
        kurtosis = np.mean(((img_float - mean_val) / 255.0)**4) - 3
        
        return {
            'edge_density_texture': float(edge_density),
            'texture_variance': float(variance),
            'texture_skewness': float(skewness),
            'texture_kurtosis': float(kurtosis)
        }
        
    def _empty_texture_features(self):
        """Return empty texture features for error cases."""
        return {
            'texture_contrast': 0.0,
            'texture_dissimilarity': 0.0,
            'texture_homogeneity': 0.0,
            'texture_energy': 0.0,
            'lbp_uniformity': 0.0,
            'lbp_entropy': 0.0,
            'gabor_response_1': 0.0,
            'gabor_response_2': 0.0,
            'gabor_response_3': 0.0,
            'gabor_response_4': 0.0,
            'edge_density_texture': 0.0,
            'texture_variance': 0.0,
            'texture_skewness': 0.0,
            'texture_kurtosis': 0.0
        }
        
    def classify_texture_type(self, features):
        """
        Classify the general texture type based on features.
        
        Returns:
            String describing the texture type
        """
        contrast = features.get('texture_contrast', 0)
        uniformity = features.get('lbp_uniformity', 0)
        energy = features.get('texture_energy', 0)
        edge_density = features.get('edge_density_texture', 0)
        
        # Smooth texture (typical of satellite bodies)
        if energy > 0.5 and uniformity > 0.3 and contrast < 0.5:
            return "smooth"
        
        # Rough/irregular texture (typical of debris)
        elif contrast > 1.0 and uniformity < 0.2:
            return "rough"
        
        # Structured texture (typical of satellites with panels)
        elif edge_density > 0.3 and 0.2 < uniformity < 0.8:
            return "structured"
        
        # Periodic texture (solar panels, regular patterns)
        elif uniformity > 0.5 and energy > 0.3:
            return "periodic"
        
        # Default
        else:
            return "complex"
