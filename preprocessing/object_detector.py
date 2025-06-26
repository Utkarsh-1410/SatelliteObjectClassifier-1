"""
Object detection and bounding box extraction for satellite images
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from utils.logger import get_logger

class ObjectDetector:
    """Detects and extracts satellite objects from images."""
    
    def __init__(self):
        self.logger = get_logger()
        
    def detect_satellite_object(self, image):
        """
        Detect the main satellite object in the image.
        
        Returns:
            tuple: (object_mask, bounding_box, confidence)
        """
        try:
            # Multiple detection strategies
            strategies = [
                self._detect_by_edges,
                self._detect_by_clustering,
                self._detect_by_intensity,
                self._detect_by_contours
            ]
            
            best_detection = None
            best_confidence = 0
            
            for strategy in strategies:
                mask, bbox, confidence = strategy(image)
                if confidence > best_confidence:
                    best_detection = (mask, bbox, confidence)
                    best_confidence = confidence
            
            return best_detection if best_detection else (None, None, 0)
            
        except Exception as e:
            self.logger.error(f"Error in object detection: {str(e)}")
            return None, None, 0
            
    def _detect_by_edges(self, image):
        """Detect object using edge information."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_FILL_HOLES, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, 0
        
        # Select best contour based on area and compactness
        best_contour = self._select_best_contour(contours, image.shape)
        
        if best_contour is None:
            return None, None, 0
        
        # Create mask and bounding box
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [best_contour], 255)
        
        bbox = cv2.boundingRect(best_contour)
        
        # Calculate confidence based on edge strength and shape
        confidence = self._calculate_edge_confidence(best_contour, edges)
        
        return mask, bbox, confidence
        
    def _detect_by_clustering(self, image):
        """Detect object using intensity clustering."""
        # Reshape image for clustering
        pixel_values = image.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Reshape labels back to image shape
        labels = labels.reshape(image.shape)
        
        # Find the cluster that represents the object
        # Assume object has medium intensity (not background or very bright)
        centers = centers.flatten()
        sorted_indices = np.argsort(centers)
        object_label = sorted_indices[1]  # Middle intensity cluster
        
        # Create mask
        mask = (labels == object_label).astype(np.uint8) * 255
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find largest connected component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, 0
        
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        bbox = cv2.boundingRect(largest_contour)
        confidence = self._calculate_clustering_confidence(mask, image)
        
        return mask, bbox, confidence
        
    def _detect_by_intensity(self, image):
        """Detect object using intensity thresholding."""
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        
        # Also try Otsu thresholding
        _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine both approaches
        combined = cv2.bitwise_and(adaptive_thresh, otsu_thresh)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, 0
        
        # Select best contour
        best_contour = self._select_best_contour(contours, image.shape)
        
        if best_contour is None:
            return None, None, 0
        
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [best_contour], 255)
        
        bbox = cv2.boundingRect(best_contour)
        confidence = self._calculate_intensity_confidence(mask, image)
        
        return mask, bbox, confidence
        
    def _detect_by_contours(self, image):
        """Detect object using contour analysis."""
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Gradient magnitude
        grad_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold gradient
        _, thresh = cv2.threshold(gradient_magnitude.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, 0
        
        # Filter contours by area and shape
        valid_contours = []
        total_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 0.01 * total_area < area < 0.8 * total_area:
                # Check if contour is roughly compact
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.1:  # Not too elongated
                        valid_contours.append(contour)
        
        if not valid_contours:
            return None, None, 0
        
        # Select contour closest to center
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        best_contour = min(valid_contours, 
                          key=lambda c: np.linalg.norm(np.array(cv2.moments(c)['m10'] / cv2.moments(c)['m00'] if cv2.moments(c)['m00'] > 0 else center_x) - center_x) +
                                       np.linalg.norm(np.array(cv2.moments(c)['m01'] / cv2.moments(c)['m00'] if cv2.moments(c)['m00'] > 0 else center_y) - center_y))
        
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [best_contour], 255)
        
        bbox = cv2.boundingRect(best_contour)
        confidence = self._calculate_contour_confidence(best_contour, image)
        
        return mask, bbox, confidence
        
    def _select_best_contour(self, contours, image_shape):
        """Select the best contour based on multiple criteria."""
        if not contours:
            return None
        
        total_area = image_shape[0] * image_shape[1]
        center_x, center_y = image_shape[1] // 2, image_shape[0] // 2
        
        scored_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip very small or very large contours
            if area < 0.005 * total_area or area > 0.8 * total_area:
                continue
            
            # Calculate contour properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Circularity (how close to a circle)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Distance from center
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
                
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            normalized_dist = dist_from_center / np.sqrt(center_x**2 + center_y**2)
            
            # Solidity (convexity)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Calculate score (higher is better)
            area_score = min(area / (0.1 * total_area), 1.0)  # Prefer medium-sized objects
            circularity_score = min(circularity * 2, 1.0)  # Prefer circular objects
            center_score = 1.0 - normalized_dist  # Prefer centered objects
            solidity_score = solidity  # Prefer convex objects
            
            total_score = (area_score * 0.3 + circularity_score * 0.2 + 
                          center_score * 0.3 + solidity_score * 0.2)
            
            scored_contours.append((contour, total_score))
        
        if not scored_contours:
            return None
        
        # Return contour with highest score
        return max(scored_contours, key=lambda x: x[1])[0]
        
    def _calculate_edge_confidence(self, contour, edges):
        """Calculate confidence based on edge strength."""
        mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Calculate edge density within the contour
        edge_pixels = np.sum(edges[mask > 0] > 0)
        total_pixels = np.sum(mask > 0)
        
        if total_pixels == 0:
            return 0
        
        edge_density = edge_pixels / total_pixels
        return min(edge_density * 2, 1.0)
        
    def _calculate_clustering_confidence(self, mask, image):
        """Calculate confidence based on clustering quality."""
        if np.sum(mask) == 0:
            return 0
        
        # Calculate intensity variance within and outside the mask
        object_pixels = image[mask > 0]
        background_pixels = image[mask == 0]
        
        if len(object_pixels) == 0 or len(background_pixels) == 0:
            return 0
        
        object_var = np.var(object_pixels)
        background_var = np.var(background_pixels)
        
        # Good segmentation should have low within-cluster variance
        # and high between-cluster difference
        mean_diff = abs(np.mean(object_pixels) - np.mean(background_pixels))
        confidence = mean_diff / (object_var + background_var + 1)
        
        return min(confidence / 50, 1.0)
        
    def _calculate_intensity_confidence(self, mask, image):
        """Calculate confidence based on intensity distribution."""
        if np.sum(mask) == 0:
            return 0
        
        # Calculate histogram separation
        object_hist = cv2.calcHist([image], [0], mask, [256], [0, 256])
        background_mask = 255 - mask
        bg_hist = cv2.calcHist([image], [0], background_mask, [256], [0, 256])
        
        # Normalize histograms
        object_hist = object_hist / (np.sum(object_hist) + 1e-6)
        bg_hist = bg_hist / (np.sum(bg_hist) + 1e-6)
        
        # Calculate histogram intersection (lower is better for separation)
        intersection = np.sum(np.minimum(object_hist, bg_hist))
        confidence = 1.0 - intersection
        
        return confidence
        
    def _calculate_contour_confidence(self, contour, image):
        """Calculate confidence based on contour properties."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0
        
        # Calculate shape properties
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Calculate solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Combine metrics
        confidence = (circularity * 0.5 + solidity * 0.5)
        return min(confidence, 1.0)
