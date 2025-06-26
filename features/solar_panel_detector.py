"""
Solar panel detection for satellite classification
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from utils.logger import get_logger

class SolarPanelDetector:
    """Detects solar panels in satellite images."""
    
    def __init__(self):
        self.logger = get_logger()
        
    def detect_solar_panels(self, image):
        """
        Detect solar panels in the image using multiple techniques.
        
        Args:
            image: Grayscale image
            
        Returns:
            Dictionary with solar panel features
        """
        try:
            # Method 1: Rectangular shape detection
            rect_panels = self._detect_rectangular_panels(image)
            
            # Method 2: Reflective surface detection
            reflective_panels = self._detect_reflective_surfaces(image)
            
            # Method 3: Texture-based detection
            texture_panels = self._detect_by_texture(image)
            
            # Combine results
            combined_panels = self._combine_detections(rect_panels, reflective_panels, texture_panels)
            
            # Calculate features
            total_area = self._calculate_panel_area(combined_panels, image.shape)
            panel_count = len(combined_panels)
            
            return {
                'solar_panel_area': float(total_area),
                'solar_panel_count': float(panel_count)
            }
            
        except Exception as e:
            self.logger.error(f"Error in solar panel detection: {str(e)}")
            return {'solar_panel_area': 0.0, 'solar_panel_count': 0.0}
            
    def _detect_rectangular_panels(self, image):
        """Detect rectangular solar panels using edge detection and contour analysis."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_panels = []
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) >= 4 and len(approx) <= 8:
                area = cv2.contourArea(contour)
                
                # Filter by area (solar panels shouldn't be too small or too large)
                total_area = image.shape[0] * image.shape[1]
                if 0.001 * total_area < area < 0.1 * total_area:
                    
                    # Check aspect ratio (solar panels are often rectangular)
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        if 1.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio for panels
                            rectangular_panels.append(contour)
        
        return rectangular_panels
        
    def _detect_reflective_surfaces(self, image):
        """Detect solar panels based on reflective properties."""
        # Solar panels often appear as bright, uniform regions
        
        # Apply Gaussian blur to smooth out noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Find bright regions (potential reflective surfaces)
        bright_threshold = np.percentile(blurred, 75)  # Top 25% brightest pixels
        bright_mask = blurred > bright_threshold
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        reflective_panels = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            total_area = image.shape[0] * image.shape[1]
            
            # Filter by area and check uniformity
            if 0.001 * total_area < area < 0.05 * total_area:
                # Check intensity uniformity within the region
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                
                region_pixels = image[mask > 0]
                if len(region_pixels) > 0:
                    intensity_std = np.std(region_pixels)
                    intensity_mean = np.mean(region_pixels)
                    
                    # Solar panels should have relatively uniform intensity
                    uniformity = 1.0 - (intensity_std / (intensity_mean + 1))
                    if uniformity > 0.7:  # High uniformity
                        reflective_panels.append(contour)
        
        return reflective_panels
        
    def _detect_by_texture(self, image):
        """Detect solar panels based on texture characteristics."""
        # Solar panels often have a distinctive grid-like texture
        
        # Apply different filters to detect grid patterns
        
        # Horizontal lines detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Vertical lines detection
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine horizontal and vertical lines
        grid_pattern = cv2.add(horizontal_lines, vertical_lines)
        
        # Threshold to get binary grid pattern
        _, grid_binary = cv2.threshold(grid_pattern, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(grid_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        texture_panels = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            total_area = image.shape[0] * image.shape[1]
            
            if 0.001 * total_area < area < 0.08 * total_area:
                # Check if the region has sufficient grid-like structure
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                
                grid_density = np.sum(grid_binary[mask > 0] > 0) / np.sum(mask > 0)
                
                if grid_density > 0.1:  # Sufficient grid structure
                    texture_panels.append(contour)
        
        return texture_panels
        
    def _combine_detections(self, rect_panels, reflective_panels, texture_panels):
        """Combine detection results from different methods."""
        all_panels = rect_panels + reflective_panels + texture_panels
        
        if not all_panels:
            return []
        
        # Remove overlapping detections
        combined_panels = []
        
        for i, panel1 in enumerate(all_panels):
            is_duplicate = False
            
            for j, panel2 in enumerate(combined_panels):
                # Calculate overlap between contours
                overlap = self._calculate_contour_overlap(panel1, panel2)
                
                if overlap > 0.5:  # More than 50% overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                combined_panels.append(panel1)
        
        return combined_panels
        
    def _calculate_contour_overlap(self, contour1, contour2):
        """Calculate overlap ratio between two contours."""
        try:
            # Create masks for both contours
            # Determine bounding box to create appropriately sized masks
            all_points = np.vstack([contour1.reshape(-1, 2), contour2.reshape(-1, 2)])
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            
            if width <= 0 or height <= 0:
                return 0
            
            # Create masks
            mask1 = np.zeros((height, width), dtype=np.uint8)
            mask2 = np.zeros((height, width), dtype=np.uint8)
            
            # Adjust contour coordinates
            contour1_adjusted = contour1.copy()
            contour1_adjusted[:, :, 0] -= x_min
            contour1_adjusted[:, :, 1] -= y_min
            
            contour2_adjusted = contour2.copy()
            contour2_adjusted[:, :, 0] -= x_min
            contour2_adjusted[:, :, 1] -= y_min
            
            cv2.fillPoly(mask1, [contour1_adjusted], 255)
            cv2.fillPoly(mask2, [contour2_adjusted], 255)
            
            # Calculate intersection and union
            intersection = np.sum((mask1 > 0) & (mask2 > 0))
            union = np.sum((mask1 > 0) | (mask2 > 0))
            
            if union == 0:
                return 0
            
            return intersection / union
            
        except Exception:
            return 0
        
    def _calculate_panel_area(self, panels, image_shape):
        """Calculate total area of detected solar panels."""
        total_area = 0
        image_total_area = image_shape[0] * image_shape[1]
        
        for panel in panels:
            area = cv2.contourArea(panel)
            total_area += area
        
        # Return as fraction of total image area
        return total_area / image_total_area
