"""
AI-Powered Speech Bubble Placement System
Advanced bubble positioning using computer vision and AI
"""

import cv2
import numpy as np
import math
from typing import Tuple, List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import os
from backend.ai_enhanced_core import face_detector, image_processor

class AIBubblePlacer:
    """Advanced AI-powered speech bubble placement"""
    
    def __init__(self):
        self.bubble_width = 200
        self.bubble_height = 94
        self.min_distance_from_face = 80
        self.quality_threshold = 0.7
        
    def place_bubble_ai(self, image_path: str, panel_coords: Tuple[int, int, int, int], 
                       lip_coords: Optional[Tuple[int, int]] = None, 
                       dialogue: str = "") -> Tuple[int, int]:
        """AI-powered bubble placement with comprehensive analysis"""
        
        # 1. Analyze image content
        content_analysis = self._analyze_image_content(image_path)
        
        # 2. Detect faces and important regions
        faces = []
        try:
            # Use basic OpenCV face detection
            img = cv2.imread(image_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                face_rects = face_cascade.detectMultiScale(gray, 1.1, 4)
                faces = [{'face_box': {'x': x, 'y': y, 'width': w, 'height': h}} for (x, y, w, h) in face_rects]
        except:
            faces = []
        
        # 3. Generate candidate positions
        candidates = self._generate_candidates(panel_coords, faces, content_analysis)
        
        # 4. Score candidates using AI
        scored_candidates = self._score_candidates(candidates, faces, content_analysis, dialogue)
        
        # 5. Select optimal position
        best_position = self._select_optimal_position(scored_candidates, lip_coords)
        
        return best_position
    
    def _analyze_image_content(self, image_path: str) -> Dict:
        """Comprehensive image content analysis"""
        img = cv2.imread(image_path)
        if img is None:
            return {'salient_regions': [], 'empty_areas': [], 'busy_areas': []}
        
        # 1. Salient region detection
        salient_regions = self._detect_salient_regions(img)
        
        # 2. Empty area detection
        empty_areas = self._detect_empty_areas(img)
        
        # 3. Busy area detection
        busy_areas = self._detect_busy_areas(img)
        
        # 4. Edge analysis
        edge_analysis = self._analyze_edges(img)
        
        return {
            'salient_regions': salient_regions,
            'empty_areas': empty_areas,
            'busy_areas': busy_areas,
            'edge_analysis': edge_analysis
        }
    
    def _detect_salient_regions(self, img: np.ndarray) -> List[Dict]:
        """Detect salient (attention-grabbing) regions"""
        # Convert to LAB color space for better saliency detection
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Calculate color contrast
        l_channel = lab[:,:,0]
        a_channel = lab[:,:,1]
        b_channel = lab[:,:,2]
        
        # Create simple saliency map using edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find regions with high edge density
        salient_regions = []
        kernel = np.ones((20, 20), np.uint8)
        edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
        
        threshold = np.percentile(edge_density, 85)
        salient_mask = edge_density > threshold
        
        # Find contours of salient regions
        contours, _ = cv2.findContours(salient_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Minimum area
                x, y, w, h = cv2.boundingRect(contour)
                salient_regions.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'area': w * h,
                    'saliency_score': np.mean(edge_density[y:y+h, x:x+w])
                })
        
        return salient_regions
    
    def _detect_empty_areas(self, img: np.ndarray) -> List[Dict]:
        """Detect areas suitable for bubble placement"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance
        kernel = np.ones((20, 20), np.float32) / 400
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        mean_sq = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = mean_sq - mean**2
        
        # Find low-variance regions (empty areas)
        threshold = np.percentile(variance, 20)
        empty_mask = variance < threshold
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(empty_mask.astype(np.uint8))
        
        empty_areas = []
        for label in range(1, num_labels):
            mask = labels == label
            if np.sum(mask) > 2000:  # Minimum area
                y_coords, x_coords = np.where(mask)
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                empty_areas.append({
                    'x': x_min, 'y': y_min, 
                    'width': x_max - x_min, 'height': y_max - y_min,
                    'area': (x_max - x_min) * (y_max - y_min),
                    'variance': np.mean(variance[y_min:y_max, x_min:x_max])
                })
        
        return empty_areas
    
    def _detect_busy_areas(self, img: np.ndarray) -> List[Dict]:
        """Detect busy/complex areas to avoid"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance
        kernel = np.ones((20, 20), np.float32) / 400
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        mean_sq = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = mean_sq - mean**2
        
        # Find high-variance regions (busy areas)
        threshold = np.percentile(variance, 80)
        busy_mask = variance > threshold
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(busy_mask.astype(np.uint8))
        
        busy_areas = []
        for label in range(1, num_labels):
            mask = labels == label
            if np.sum(mask) > 1000:  # Minimum area
                y_coords, x_coords = np.where(mask)
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                busy_areas.append({
                    'x': x_min, 'y': y_min, 
                    'width': x_max - x_min, 'height': y_max - y_min,
                    'area': (x_max - x_min) * (y_max - y_min),
                    'variance': np.mean(variance[y_min:y_max, x_min:x_max])
                })
        
        return busy_areas
    
    def _analyze_edges(self, img: np.ndarray) -> Dict:
        """Analyze edge distribution for optimal placement"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale edge detection
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 100, 200)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Calculate edge density in different regions
        height, width = edges.shape
        regions = {
            'top_left': edges[0:height//2, 0:width//2],
            'top_right': edges[0:height//2, width//2:],
            'bottom_left': edges[height//2:, 0:width//2],
            'bottom_right': edges[height//2:, width//2:]
        }
        
        edge_density = {}
        for region_name, region in regions.items():
            edge_density[region_name] = np.sum(region > 0) / region.size
        
        return {
            'edge_density': edge_density,
            'total_edge_density': np.sum(edges > 0) / edges.size
        }
    
    def _generate_candidates(self, panel_coords: Tuple[int, int, int, int], 
                           faces: List[Dict], content_analysis: Dict) -> List[Dict]:
        """Generate candidate positions for bubble placement"""
        left, right, top, bottom = panel_coords
        panel_width = right - left
        panel_height = bottom - top
        
        candidates = []
        
        # 1. Corner positions (high priority)
        corner_margin = 40
        corners = [
            (left + corner_margin, top + corner_margin),
            (right - corner_margin - self.bubble_width, top + corner_margin),
            (left + corner_margin, bottom - corner_margin - self.bubble_height),
            (right - corner_margin - self.bubble_width, bottom - corner_margin - self.bubble_height)
        ]
        
        for x, y in corners:
            candidates.append({
                'x': x, 'y': y,
                'type': 'corner',
                'priority': 0.9
            })
        
        # 2. Edge positions
        edge_margin = 30
        edge_positions = []
        
        # Top edge
        for x in range(left + edge_margin, right - edge_margin - self.bubble_width, 50):
            edge_positions.append((x, top + edge_margin))
        
        # Right edge
        for y in range(top + edge_margin, bottom - edge_margin - self.bubble_height, 50):
            edge_positions.append((right - edge_margin - self.bubble_width, y))
        
        for x, y in edge_positions:
            candidates.append({
                'x': x, 'y': y,
                'type': 'edge',
                'priority': 0.7
            })
        
        # 3. Empty area centers
        for area in content_analysis['empty_areas']:
            center_x = area['x'] + area['width'] // 2
            center_y = area['y'] + area['height'] // 2
            
            # Check if bubble fits in this area
            if (area['width'] >= self.bubble_width and 
                area['height'] >= self.bubble_height):
                candidates.append({
                    'x': center_x - self.bubble_width // 2,
                    'y': center_y - self.bubble_height // 2,
                    'type': 'empty_area',
                    'priority': 0.8,
                    'area_quality': area['variance']
                })
        
        # 4. Upper region positions (preferred for dialogue)
        upper_y = top + panel_height * 0.2
        for x in range(left + 50, right - 50 - self.bubble_width, 80):
            candidates.append({
                'x': x, 'y': upper_y,
                'type': 'upper_region',
                'priority': 0.85
            })
        
        return candidates
    
    def _score_candidates(self, candidates: List[Dict], faces: List[Dict], 
                         content_analysis: Dict, dialogue: str) -> List[Dict]:
        """Score candidates using AI and heuristics"""
        scored_candidates = []
        
        for candidate in candidates:
            score = candidate['priority']  # Base score
            
            # 1. Face avoidance scoring
            face_penalty = self._calculate_face_penalty(candidate, faces)
            score -= face_penalty
            
            # 2. Content analysis scoring
            content_score = self._calculate_content_score(candidate, content_analysis)
            score += content_score
            
            # 3. Dialogue-specific scoring
            dialogue_score = self._calculate_dialogue_score(candidate, dialogue, content_analysis)
            score += dialogue_score
            
            # 4. Edge density scoring
            edge_score = self._calculate_edge_score(candidate, content_analysis)
            score += edge_score
            
            # 5. Position preference scoring
            position_score = self._calculate_position_score(candidate)
            score += position_score
            
            # Ensure score is within bounds
            score = max(0.0, min(1.0, score))
            
            candidate['final_score'] = score
            scored_candidates.append(candidate)
        
        return scored_candidates
    
    def _calculate_face_penalty(self, candidate: Dict, faces: List[Dict]) -> float:
        """Calculate penalty for being too close to faces"""
        penalty = 0.0
        
        for face in faces:
            face_center_x = face['face_box']['x'] + face['face_box']['width'] // 2
            face_center_y = face['face_box']['y'] + face['face_box']['height'] // 2
            
            bubble_center_x = candidate['x'] + self.bubble_width // 2
            bubble_center_y = candidate['y'] + self.bubble_height // 2
            
            distance = math.sqrt((bubble_center_x - face_center_x)**2 + 
                               (bubble_center_y - face_center_y)**2)
            
            if distance < self.min_distance_from_face:
                penalty += (self.min_distance_from_face - distance) / self.min_distance_from_face
        
        return min(0.5, penalty)  # Cap penalty at 0.5
    
    def _calculate_content_score(self, candidate: Dict, content_analysis: Dict) -> float:
        """Calculate score based on content analysis"""
        score = 0.0
        
        # Check if candidate overlaps with busy areas
        for busy_area in content_analysis['busy_areas']:
            if self._rectangles_overlap(
                (candidate['x'], candidate['y'], self.bubble_width, self.bubble_height),
                (busy_area['x'], busy_area['y'], busy_area['width'], busy_area['height'])
            ):
                score -= 0.3  # Penalty for overlapping busy areas
        
        # Bonus for being in empty areas
        for empty_area in content_analysis['empty_areas']:
            if self._rectangles_overlap(
                (candidate['x'], candidate['y'], self.bubble_width, self.bubble_height),
                (empty_area['x'], empty_area['y'], empty_area['width'], empty_area['height'])
            ):
                score += 0.2  # Bonus for being in empty areas
        
        return score
    
    def _calculate_dialogue_score(self, candidate: Dict, dialogue: str, 
                                 content_analysis: Dict) -> float:
        """Calculate score based on dialogue content"""
        score = 0.0
        
        # Short dialogue prefers upper positions
        if len(dialogue) < 50:
            if candidate['y'] < 100:  # Upper region
                score += 0.1
        
        # Long dialogue prefers larger empty areas
        if len(dialogue) > 100:
            for empty_area in content_analysis['empty_areas']:
                if self._rectangles_overlap(
                    (candidate['x'], candidate['y'], self.bubble_width, self.bubble_height),
                    (empty_area['x'], empty_area['y'], empty_area['width'], empty_area['height'])
                ):
                    if empty_area['area'] > 10000:  # Large empty area
                        score += 0.15
        
        return score
    
    def _calculate_edge_score(self, candidate: Dict, content_analysis: Dict) -> float:
        """Calculate score based on edge density"""
        score = 0.0
        
        edge_density = content_analysis['edge_analysis']['edge_density']
        
        # Prefer regions with lower edge density
        bubble_center_x = candidate['x'] + self.bubble_width // 2
        bubble_center_y = candidate['y'] + self.bubble_height // 2
        
        # Determine which region the bubble is in
        if bubble_center_x < 400 and bubble_center_y < 300:  # Top-left
            region_density = edge_density['top_left']
        elif bubble_center_x >= 400 and bubble_center_y < 300:  # Top-right
            region_density = edge_density['top_right']
        elif bubble_center_x < 400 and bubble_center_y >= 300:  # Bottom-left
            region_density = edge_density['bottom_left']
        else:  # Bottom-right
            region_density = edge_density['bottom_right']
        
        # Lower edge density is better
        score += (1.0 - region_density) * 0.2
        
        return score
    
    def _calculate_position_score(self, candidate: Dict) -> float:
        """Calculate score based on position preferences"""
        score = 0.0
        
        # Prefer upper positions for dialogue
        if candidate['type'] == 'upper_region':
            score += 0.1
        
        # Prefer corners for short dialogue
        if candidate['type'] == 'corner':
            score += 0.05
        
        return score
    
    def _rectangles_overlap(self, rect1: Tuple[int, int, int, int], 
                           rect2: Tuple[int, int, int, int]) -> bool:
        """Check if two rectangles overlap"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or 
                   y1 + h1 <= y2 or y2 + h2 <= y1)
    
    def _select_optimal_position(self, scored_candidates: List[Dict], 
                                lip_coords: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        """Select the optimal position from scored candidates"""
        if not scored_candidates:
            return (100, 100)  # Default position
        
        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Select the best candidate
        best_candidate = scored_candidates[0]
        
        # Apply final adjustments based on lip coordinates
        if lip_coords and lip_coords != (-1, -1):
            final_x, final_y = self._adjust_for_lip_position(
                best_candidate['x'], best_candidate['y'], lip_coords
            )
        else:
            final_x, final_y = best_candidate['x'], best_candidate['y']
        
        return (final_x, final_y)
    
    def _adjust_for_lip_position(self, x: int, y: int, lip_coords: Tuple[int, int]) -> Tuple[int, int]:
        """Make final adjustments based on lip position"""
        lip_x, lip_y = lip_coords
        
        # If bubble is too close to lip, move it away
        distance = math.sqrt((x - lip_x)**2 + (y - lip_y)**2)
        
        if distance < self.min_distance_from_face:
            # Calculate direction away from lip
            dx = x - lip_x
            dy = y - lip_y
            
            if dx == 0 and dy == 0:
                dx, dy = 1, 0  # Default direction
            
            # Normalize and move away
            magnitude = math.sqrt(dx**2 + dy**2)
            dx = dx / magnitude * self.min_distance_from_face
            dy = dy / magnitude * self.min_distance_from_face
            
            x = lip_x + dx
            y = lip_y + dy
        
        return (int(x), int(y))

# Global instance
ai_bubble_placer = AIBubblePlacer()