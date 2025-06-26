"""
Specialized noise handling for satellite images
"""

import numpy as np
import cv2
from scipy import signal, ndimage
from utils.logger import get_logger

class NoiseHandler:
    """Handles noise addition and removal for satellite images."""
    
    def __init__(self):
        self.logger = get_logger()
        
    def add_space_environment_noise(self, image):
        """Add realistic space environment noise."""
        noisy = image.copy().astype(np.float32)
        
        # Cosmic ray noise (random bright pixels)
        cosmic_rays = np.random.random(image.shape) < 0.001
        noisy[cosmic_rays] = 255
        
        # Thermal noise from sensors
        thermal_noise = np.random.normal(0, 3, image.shape)
        noisy += thermal_noise
        
        # Quantization noise
        noisy = np.round(noisy / 4) * 4  # Simulate 6-bit quantization
        
        # Atmospheric scattering (for low orbit objects)
        if np.random.random() < 0.3:  # 30% chance of atmospheric effects
            scattering_kernel = self._create_atmospheric_kernel()
            noisy = signal.convolve2d(noisy, scattering_kernel, mode='same', boundary='symm')
        
        return np.clip(noisy, 0, 255).astype(np.uint8)
        
    def _create_atmospheric_kernel(self):
        """Create atmospheric scattering kernel."""
        size = 5
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                distance = np.sqrt((i - center)**2 + (j - center)**2)
                kernel[i, j] = np.exp(-distance**2 / (2 * 1.5**2))
        
        return kernel / np.sum(kernel)
        
    def advanced_denoise(self, image):
        """Advanced denoising specifically for satellite images."""
        # Start with the original image
        denoised = image.copy()
        
        # 1. Remove cosmic ray hits (isolated bright pixels)
        denoised = self._remove_cosmic_rays(denoised)
        
        # 2. Wavelet denoising for preserving edges
        denoised = self._wavelet_denoise(denoised)
        
        # 3. Anisotropic diffusion for edge preservation
        denoised = self._anisotropic_diffusion(denoised)
        
        # 4. Non-local means for texture preservation
        denoised = cv2.fastNlMeansDenoising(denoised, None, 8, 7, 21)
        
        return denoised
        
    def _remove_cosmic_rays(self, image):
        """Remove cosmic ray hits (isolated bright pixels)."""
        # Use median filter to detect outliers
        median_filtered = cv2.medianBlur(image, 3)
        diff = np.abs(image.astype(np.float32) - median_filtered.astype(np.float32))
        
        # Threshold for cosmic ray detection
        threshold = np.mean(diff) + 3 * np.std(diff)
        cosmic_ray_mask = diff > threshold
        
        # Replace cosmic rays with median values
        result = image.copy()
        result[cosmic_ray_mask] = median_filtered[cosmic_ray_mask]
        
        return result
        
    def _wavelet_denoise(self, image):
        """Simple wavelet-like denoising using Gaussian pyramids."""
        # Create Gaussian pyramid
        pyramid = [image.astype(np.float32)]
        
        for i in range(3):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        
        # Denoise each level
        for i in range(len(pyramid)):
            # Apply gentle Gaussian blur
            pyramid[i] = cv2.GaussianBlur(pyramid[i], (3, 3), 0.5)
        
        # Reconstruct image
        result = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            result = cv2.pyrUp(result)
            # Ensure same size as target level
            if result.shape != pyramid[i].shape:
                result = cv2.resize(result, (pyramid[i].shape[1], pyramid[i].shape[0]))
            
            # Blend with original level
            result = 0.7 * pyramid[i] + 0.3 * result
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    def _anisotropic_diffusion(self, image, iterations=10, delta_t=0.2, kappa=20):
        """Anisotropic diffusion for edge-preserving smoothing."""
        img = image.astype(np.float32)
        
        for _ in range(iterations):
            # Calculate gradients
            grad_n = np.zeros_like(img)
            grad_s = np.zeros_like(img)
            grad_e = np.zeros_like(img)
            grad_w = np.zeros_like(img)
            
            grad_n[:-1, :] = img[1:, :] - img[:-1, :]
            grad_s[1:, :] = img[:-1, :] - img[1:, :]
            grad_e[:, :-1] = img[:, 1:] - img[:, :-1]
            grad_w[:, 1:] = img[:, :-1] - img[:, 1:]
            
            # Calculate diffusion coefficients
            c_n = np.exp(-(grad_n / kappa)**2)
            c_s = np.exp(-(grad_s / kappa)**2)
            c_e = np.exp(-(grad_e / kappa)**2)
            c_w = np.exp(-(grad_w / kappa)**2)
            
            # Update image
            img += delta_t * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
        
        return np.clip(img, 0, 255).astype(np.uint8)
