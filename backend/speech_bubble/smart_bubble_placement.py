#!/usr/bin/env python3
"""
Smart Bubble Placement System
Uses image analysis to find optimal bubble positions without CAM data
"""

import cv2
import numpy as np
import math
from typing import Tuple, List, Optional
from backend.utils import get_panel_type, types

BUBBLE_WIDTH = 200
BUBBLE_HEIGHT = 94

class SmartBubblePlacer:
    def __init__(self):
        """Initialize smart bubble placement system"""
        self.face_detector = None
        try:
            from backend.speech_bubble.modern_face_detection import ModernFaceDetector
            self.face_detector = ModernFaceDetector()
        except ImportError:
            print("Modern face detector not available, using basic placement")
    
    def analyze_image_content(self, image_path: str) -> dict:
        """
        Analyze image content to find optimal bubble placement areas
        Returns: {
            'face_regions': [(x, y, w, h), ...],
            'empty_areas': [(x, y, w, h), ...],
            'busy_areas': [(x, y, w, h), ...],
            'edges': [(x, y), ...]
        }
        """
        image = cv2.imread(image_path)
        if image is None:
            return {'face_regions': [], 'empty_areas': [], 'busy_areas': [], 'edges': []}
        
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect faces
        face_regions = self._detect_faces(image)
        
        # 2. Find empty areas (low variance regions)
        empty_areas = self._find_empty_areas(gray)
        
        # 3. Find busy areas (high variance regions)
        busy_areas = self._find_busy_areas(gray)
        
        # 4. Find edge positions
        edges = self._find_edge_positions(width, height)
        
        return {
            'face_regions': face_regions,
            'empty_areas': empty_areas,
            'busy_areas': busy_areas,
            'edges': edges
        }
    
    def _detect_faces(self, image) -> List[Tuple[int, int, int, int]]:
        """Detect face regions in image"""
        if self.face_detector:
            # Use modern face detector with image object
            try:
                # Convert image to format expected by face detector
                if isinstance(image, str):
                    # If it's a file path, read the image
                    img = cv2.imread(image)
                else:
                    # If it's already an image object
                    img = image
                
                faces = self.face_detector.detect_faces_opencv(img)
                face_regions = []
                for face in faces:
                    if face != (-1, -1):
                        # Create face region around detected point
                        x, y = face
                        face_regions.append((x-50, y-50, 100, 100))
                return face_regions
            except Exception as e:
                print(f"Face detection error: {e}")
                return []
        else:
            # Fallback to basic face detection
            try:
                if isinstance(image, str):
                    img = cv2.imread(image)
                else:
                    img = image
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                return [(x, y, w, h) for (x, y, w, h) in faces]
            except Exception as e:
                print(f"Fallback face detection error: {e}")
                return []
    
    def _find_empty_areas(self, gray_image) -> List[Tuple[int, int, int, int]]:
        """Find areas with low variance (good for bubbles)"""
        # Calculate local variance
        kernel = np.ones((20, 20), np.float32) / 400
        mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        mean_sq = cv2.filter2D((gray_image.astype(np.float32))**2, -1, kernel)
        variance = mean_sq - mean**2
        
        # Find low variance regions
        threshold = np.percentile(variance, 20)  # Bottom 20% variance
        low_var_mask = variance < threshold
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(low_var_mask.astype(np.uint8))
        
        empty_areas = []
        for label in range(1, num_labels):
            mask = labels == label
            if np.sum(mask) > 1000:  # Minimum area
                y_coords, x_coords = np.where(mask)
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                empty_areas.append((x_min, y_min, x_max-x_min, y_max-y_min))
        
        return empty_areas
    
    def _find_busy_areas(self, gray_image) -> List[Tuple[int, int, int, int]]:
        """Find areas with high variance (avoid for bubbles)"""
        # Calculate local variance
        kernel = np.ones((20, 20), np.float32) / 400
        mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        mean_sq = cv2.filter2D((gray_image.astype(np.float32))**2, -1, kernel)
        variance = mean_sq - mean**2
        
        # Find high variance regions
        threshold = np.percentile(variance, 80)  # Top 20% variance
        high_var_mask = variance > threshold
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(high_var_mask.astype(np.uint8))
        
        busy_areas = []
        for label in range(1, num_labels):
            mask = labels == label
            if np.sum(mask) > 500:  # Minimum area
                y_coords, x_coords = np.where(mask)
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                busy_areas.append((x_min, y_min, x_max-x_min, y_max-y_min))
        
        return busy_areas
    
    def _find_edge_positions(self, width: int, height: int) -> List[Tuple[int, int]]:
        """Find good edge positions for bubbles"""
        margin = 50
        edge_positions = []
        
        # Top edge
        for x in range(margin, width - margin, 100):
            edge_positions.append((x, margin))
        
        # Right edge
        for y in range(margin, height - margin, 100):
            edge_positions.append((width - margin, y))
        
        # Top-right corner area
        corner_margin = 80
        for x in range(width - corner_margin - 100, width - corner_margin, 50):
            for y in range(margin, margin + 100, 50):
                edge_positions.append((x, y))
        
        return edge_positions
    
    def get_optimal_bubble_position(self, image_path: str, panel_coords: Tuple[int, int, int, int], 
                                  lip_coords: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Find optimal bubble position based on image analysis
        """
        # Analyze image content
        analysis = self.analyze_image_content(image_path)
        
        # Get panel dimensions
        left, right, top, bottom = panel_coords
        panel_width = right - left
        panel_height = bottom - top
        
        # Generate candidate positions
        candidates = []
        
        # 1. Edge positions (highest priority)
        for edge_x, edge_y in analysis['edges']:
            if (left <= edge_x <= right and top <= edge_y <= bottom):
                candidates.append((edge_x, edge_y, 100))  # High score
        
        # 2. Empty areas (good for bubbles)
        for x, y, w, h in analysis['empty_areas']:
            center_x = x + w // 2
            center_y = y + h // 2
            if (left <= center_x <= right and top <= center_y <= bottom):
                candidates.append((center_x, center_y, 80))  # Good score
        
        # 3. Upper area positions (preferred)
        upper_y = top + panel_height * 0.2  # 20% from top
        for x in range(left + 50, right - 50, 100):
            candidates.append((x, upper_y, 70))  # Medium score
        
        # 4. Corner positions
        corner_margin = 40
        corners = [
            (left + corner_margin, top + corner_margin),
            (right - corner_margin, top + corner_margin),
            (left + corner_margin, top + panel_height * 0.3),
            (right - corner_margin, top + panel_height * 0.3)
        ]
        for x, y in corners:
            candidates.append((x, y, 60))  # Lower score
        
        # Filter out positions that overlap with faces or busy areas
        filtered_candidates = []
        for x, y, score in candidates:
            # Check if position overlaps with face regions
            overlaps_face = False
            for fx, fy, fw, fh in analysis['face_regions']:
                if (fx <= x <= fx + fw and fy <= y <= fy + fh):
                    overlaps_face = True
                    break
            
            # Check if position overlaps with busy areas
            overlaps_busy = False
            for bx, by, bw, bh in analysis['busy_areas']:
                if (bx <= x <= bx + bw and by <= y <= by + bh):
                    overlaps_busy = True
                    break
            
            # Check distance from lip if provided
            too_close_to_lip = False
            if lip_coords and lip_coords != (-1, -1):
                lip_x, lip_y = lip_coords
                distance = math.sqrt((x - lip_x)**2 + (y - lip_y)**2)
                if distance < 80:  # 80px minimum distance
                    too_close_to_lip = True
            
            if not overlaps_face and not overlaps_busy and not too_close_to_lip:
                filtered_candidates.append((x, y, score))
        
        # Select best position
        if filtered_candidates:
            # Sort by score (highest first)
            filtered_candidates.sort(key=lambda x: x[2], reverse=True)
            best_x, best_y, _ = filtered_candidates[0]
            return (best_x, best_y)
        else:
            # Fallback to upper center
            return (left + panel_width // 2, top + panel_height * 0.2)

def get_smart_bubble_position(image_path: str, panel_coords: Tuple[int, int, int, int], 
                            lip_coords: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """Main function to get smart bubble position"""
    placer = SmartBubblePlacer()
    return placer.get_optimal_bubble_position(image_path, panel_coords, lip_coords)