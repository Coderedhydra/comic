"""
AI-Powered Speech Bubble Placement System
Simplified and robust bubble positioning.
"""

import cv2
from typing import Tuple, Optional

class AIBubblePlacer:
    """
    AIBubblePlacer finds the best position for a speech bubble.
    This version uses a simpler, more reliable heuristic-based approach.
    """
    
    def __init__(self):
        # These values are based on a panel size of 300x200
        self.bubble_width = 160
        self.bubble_height = 80
        self.panel_width = 300
        self.panel_height = 200
        self.padding = 10 # Minimum distance from the panel edge
        
    def place_bubble_ai(self, image_path: str, lip_coords: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Determines the optimal placement for a speech bubble.

        The strategy is:
        1. If a face is detected, try to place the bubble above the face.
        2. If that's not possible, try other corners (top-left, top-right).
        3. If no face is found, analyze the image for the quietest corner.
        4. Always ensure the bubble stays within the panel boundaries.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return (50, 20) # Fallback

            # If a primary speaker is identified (lip_coords are valid)
            if lip_coords and lip_coords != (-1, -1):
                lip_x, lip_y = lip_coords

                # --- Primary Strategy: Place bubble ABOVE the speaker's head ---
                # Center the bubble horizontally over the lips
                ideal_x = lip_x - (self.bubble_width // 2)
                # Place it well above the lips to clear the head
                ideal_y = lip_y - self.bubble_height - 40 # 40px buffer

                # Check if this position is valid (within panel bounds)
                if ideal_y > self.padding:
                    final_x = self._clamp(ideal_x, self.padding, self.panel_width - self.bubble_width - self.padding)
                    final_y = self._clamp(ideal_y, self.padding, self.panel_height - self.bubble_height - self.padding)
                    return (int(final_x), int(final_y))

            # --- Fallback Strategy: Find the best corner if no face or space above is poor ---
            return self._find_best_corner(image)

        except Exception as e:
            print(f"ERROR in AI bubble placer: {e}")
            return (50, 20) # Final fallback

    def _clamp(self, value, min_value, max_value):
        """Helper function to keep a value within a specific range."""
        return max(min_value, min(value, max_value))

    def _get_region_clarity(self, image, rect):
        """Calculates the 'clarity' of a region (low edge count is clearer)."""
        x, y, w, h = rect
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return float('inf') # Invalid region
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 100, 200)
        return np.sum(edges == 0) # Return count of non-edge pixels

    def _find_best_corner(self, image):
        """Analyzes the four corners of the image to find the least busy one."""
        h, w, _ = image.shape
        
        # Define the four corner regions where a bubble could go
        corner_regions = {
            "top_left": (self.padding, self.padding, self.bubble_width, self.bubble_height),
            "top_right": (w - self.bubble_width - self.padding, self.padding, self.bubble_width, self.bubble_height),
            "bottom_left": (self.padding, h - self.bubble_height - self.padding, self.bubble_width, self.bubble_height),
            "bottom_right": (w - self.bubble_width - self.padding, h - self.bubble_height - self.padding, self.bubble_width, self.bubble_height)
        }
        
        best_corner_name = None
        max_clarity = -1

        for name, rect in corner_regions.items():
            clarity = self._get_region_clarity(image, rect)
            if clarity > max_clarity:
                max_clarity = clarity
                best_corner_name = name
        
        # Return the top-left coordinate of the best corner
        best_rect = corner_regions.get(best_corner_name, ("top_left", (self.padding, self.padding)))
        return (best_rect[0], best_rect[1])

# Global instance
ai_bubble_placer = AIBubblePlacer()
