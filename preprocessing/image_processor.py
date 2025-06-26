"""
Image preprocessing pipeline for satellite object classification
"""

import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
import os
from utils.logger import get_logger

class ImageProcessor:
    """Handles all image preprocessing operations."""
    
    def __init__(self):
        self.logger = get_logger()
        
    def process_image(self, image_path, apply_noise=True, remove_background=True):
        """
        Complete image processing pipeline.
        
        Args:
            image_path: Path to the input image
            apply_noise: Whether to apply and remove noise
            remove_background: Whether to remove background
            
        Returns:
            Processed image as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size for consistency
            gray = cv2.resize(gray, (256, 256))
            
            # Apply noise and denoising if requested
            if apply_noise:
                gray = self.add_environmental_noise(gray)
                gray = self.denoise_image(gray)
            
            # Remove background if requested
            if remove_background:
                gray = self.remove_background(gray)
            
            # Detect and crop to bounding box
            gray = self.crop_to_bounding_box(gray)
            
            # Normalize intensity
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            return gray.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            raise
            
    def add_environmental_noise(self, image):
        """Add realistic environmental noise to the image."""
        noisy = image.copy().astype(np.float32)
        
        # Gaussian noise (sensor noise)
        gaussian_noise = np.random.normal(0, 5, image.shape)
        noisy += gaussian_noise
        
        # Poisson noise (photon noise)
        if np.max(noisy) > 0:
            noisy = np.random.poisson(noisy * 0.1) / 0.1
        
        # Salt and pepper noise (transmission errors)
        salt_pepper = np.random.random(image.shape)
        noisy[salt_pepper < 0.01] = 255  # Salt
        noisy[salt_pepper > 0.99] = 0    # Pepper
        
        # Atmospheric turbulence (slight blur)
        kernel_size = np.random.randint(3, 6)
        if kernel_size % 2 == 0:
            kernel_size += 1
        noisy = cv2.GaussianBlur(noisy, (kernel_size, kernel_size), 0.5)
        
        return np.clip(noisy, 0, 255).astype(np.uint8)
        
    def denoise_image(self, image):
        """Remove noise from the image using advanced denoising."""
        try:
            # Validate input
            if image is None or image.size == 0:
                return image
            
            # Ensure proper data type and range
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            denoised = image.copy()
            
            # Use bilateral filter first (more stable)
            denoised = cv2.bilateralFilter(denoised, 5, 50, 50)
            
            # Apply Gaussian blur for additional smoothing
            denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
            
            # Try Non-local means denoising with safer parameters
            try:
                denoised = cv2.fastNlMeansDenoising(denoised, None, 3, 7, 21)
            except cv2.error:
                # Fallback to median filter if Non-local means fails
                denoised = cv2.medianBlur(denoised, 3)
            
            # Morphological opening to remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
            
            return denoised
            
        except Exception as e:
            self.logger.warning(f"Denoising failed, using original image: {e}")
            return image
        
    def remove_background(self, image):
        """Remove background using advanced segmentation."""
        # Use GrabCut algorithm for background removal
        height, width = image.shape
        
        # Create initial mask
        mask = np.zeros((height, width), np.uint8)
        
        # Define rectangle around the center (assuming object is centered)
        margin = min(width, height) // 8
        rect = (margin, margin, width - 2*margin, height - 2*margin)
        
        # Initialize foreground and background models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Convert to 3-channel for GrabCut
        image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Apply GrabCut
        cv2.grabCut(image_3ch, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create final mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask
        result = image * mask2
        
        # If GrabCut fails, use Otsu thresholding as fallback
        if np.sum(mask2) < 0.1 * image.size:
            _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return result
        
    def crop_to_bounding_box(self, image):
        """Crop image to tight bounding box around the object."""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        
        # Crop image
        cropped = image[y:y+h, x:x+w]
        
        # Resize back to standard size
        if cropped.size > 0:
            cropped = cv2.resize(cropped, (128, 128))
            return cropped
        
        return cv2.resize(image, (128, 128))
        
    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
