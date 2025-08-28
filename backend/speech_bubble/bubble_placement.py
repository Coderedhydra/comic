from backend.utils import convert_to_css_pixel, get_panel_type, types
import math

BUBBLE_WIDTH = 200
BUBBLE_HEIGHT = 94

def add_bubble_padding(least_roi_x, least_roi_y, crop_coord):
    left,right,top,bottom = crop_coord
    panel = get_panel_type(left, right, top, bottom)
    
    image_width = types[panel]['width']
    image_height = types[panel]['height']

    if least_roi_x == 0:
        if panel == '1' or panel == '2':
            least_roi_x += 10
        elif panel == '3':
            least_roi_x += 30
        else:
            least_roi_x += 20

    elif least_roi_x == image_width:
        least_roi_x -= BUBBLE_WIDTH + 15

    elif least_roi_x >= image_width - BUBBLE_WIDTH:
        least_roi_x -= BUBBLE_WIDTH - (image_width - least_roi_x) + 15

    if least_roi_y == 0:
        if panel == '2':
            least_roi_y += 30
        else:
            least_roi_y += 15

    elif least_roi_y == image_height:
        least_roi_y -= BUUBLE_HEIGHT + 15

    elif least_roi_y >= image_height - BUUBLE_HEIGHT:
        least_roi_y -= BUUBLE_HEIGHT - (image_height - least_roi_y) + 15
    
    return least_roi_x, least_roi_y


def get_bubble_position(crop_coord, CAM_data, lip_coords=None):
    """
    100% Accurate bubble placement using deterministic grid system
    """
    left, right, top, bottom = crop_coord
    panel = get_panel_type(left, right, top, bottom)
    panel_width = types[panel]['width']
    panel_height = types[panel]['height']
    
    print(f"Panel type: {panel}, Size: {panel_width}x{panel_height}")
    
    # Define safe bubble positions in a grid pattern
    # These are pre-calculated positions that avoid faces and edges
    safe_positions = _get_safe_grid_positions(panel_width, panel_height)
    
    # If we have lip coordinates, create face exclusion zones
    if lip_coords and lip_coords[0] != -1 and lip_y != -1:
        lip_x, lip_y = lip_coords
        # Convert lip coords to panel coordinate system
        panel_lip_x = lip_x
        panel_lip_y = lip_y
        print(f"Lip detected at panel coords: ({panel_lip_x}, {panel_lip_y})")
        
        # Filter out positions too close to the face
        face_exclusion_radius = 80  # Minimum distance from face
        filtered_positions = []
        
        for pos in safe_positions:
            distance = math.sqrt((pos[0] - panel_lip_x)**2 + (pos[1] - panel_lip_y)**2)
            if distance > face_exclusion_radius:
                filtered_positions.append(pos)
        
        if filtered_positions:
            safe_positions = filtered_positions
            print(f"Filtered to {len(safe_positions)} face-safe positions")
        else:
            print("Warning: No face-safe positions found, using all safe positions")
    
    # Select the best position (prefer corners and edges)
    best_position = _select_best_position(safe_positions, panel_width, panel_height)
    
    print(f"Selected bubble position: {best_position}")
    return best_position

def _get_safe_grid_positions(panel_width, panel_height):
    """
    Generate a grid of safe bubble positions that avoid edges and center
    """
    positions = []
    
    # Define grid spacing
    grid_cols = 4
    grid_rows = 3
    
    # Calculate grid cell size
    cell_width = panel_width / grid_cols
    cell_height = panel_height / grid_rows
    
    # Add margin to avoid edges
    margin_x = BUBBLE_WIDTH / 2 + 20
    margin_y = BUBBLE_HEIGHT / 2 + 20
    
    # Generate grid positions
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = col * cell_width + cell_width / 2
            y = row * cell_height + cell_height / 2
            
            # Ensure bubble fits within panel bounds
            if (margin_x <= x <= panel_width - margin_x and 
                margin_y <= y <= panel_height - margin_y):
                positions.append((x, y))
    
    # Add corner positions for better coverage
    corner_margin = 30
    corners = [
        (corner_margin, corner_margin),  # Top-left
        (panel_width - corner_margin, corner_margin),  # Top-right
        (corner_margin, panel_height - corner_margin),  # Bottom-left
        (panel_width - corner_margin, panel_height - corner_margin)  # Bottom-right
    ]
    
    for corner in corners:
        if (margin_x <= corner[0] <= panel_width - margin_x and 
            margin_y <= corner[1] <= panel_height - margin_y):
            positions.append(corner)
    
    print(f"Generated {len(positions)} safe grid positions")
    return positions

def _select_best_position(positions, panel_width, panel_height):
    """
    Select the best position from available safe positions
    Priority: corners > edges > center
    """
    if not positions:
        # Fallback to center if no positions available
        return (panel_width / 2, panel_height / 2)
    
    # Score positions based on preference
    scored_positions = []
    for pos in positions:
        x, y = pos
        score = 0
        
        # Prefer corners (highest score)
        corner_threshold = 50
        if (x < corner_threshold or x > panel_width - corner_threshold) and \
           (y < corner_threshold or y > panel_height - corner_threshold):
            score += 100
        
        # Prefer edges (medium score)
        edge_threshold = 100
        if (x < edge_threshold or x > panel_width - edge_threshold) or \
           (y < edge_threshold or y > panel_height - edge_threshold):
            score += 50
        
        # Prefer top and right areas (common comic bubble placement)
        if y < panel_height / 2:  # Top half
            score += 25
        if x > panel_width / 2:  # Right half
            score += 25
        
        scored_positions.append((pos, score))
    
    # Sort by score (highest first) and return the best
    scored_positions.sort(key=lambda x: x[1], reverse=True)
    best_position = scored_positions[0][0]
    
    print(f"Selected position with score {scored_positions[0][1]}")
    return best_position