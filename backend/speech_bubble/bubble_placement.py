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


def get_bubble_position(image_coords, CAM_data=None, lip_coords=None):
    """
    Redesigned bubble placement for smart resize - positions relative to actual image content
    """
    left, right, top, bottom = image_coords
    
    # Calculate image dimensions within panel
    image_width = right - left
    image_height = bottom - top
    
    print(f"Image area: {image_width:.0f}x{image_height:.0f} at ({left:.0f}, {top:.0f})")
    
    # Define safe bubble positions relative to the actual image content
    safe_positions = _get_safe_image_positions(left, right, top, bottom, image_width, image_height)
    
    # If we have lip coordinates, create face exclusion zones
    if lip_coords and lip_coords[0] != -1 and lip_coords[1] != -1:
        lip_x, lip_y = lip_coords
        # Lip coordinates are already in panel coordinate system
        print(f"Lip detected at coords: ({lip_x}, {lip_y})")
        
        # Filter out positions too close to the face
        face_exclusion_radius = 60  # Standard exclusion radius
        filtered_positions = []
        
        for pos in safe_positions:
            distance = math.sqrt((pos[0] - lip_x)**2 + (pos[1] - lip_y)**2)
            if distance > face_exclusion_radius:
                filtered_positions.append(pos)
        
        if filtered_positions:
            safe_positions = filtered_positions
            print(f"Filtered to {len(safe_positions)} face-safe positions")
        else:
            print("Warning: No face-safe positions found, using all safe positions")
    
    # Select the best position (prefer corners and edges of image)
    best_position = _select_best_image_position(safe_positions, left, right, top, bottom)
    
    print(f"Selected bubble position: {best_position}")
    return best_position

def _get_safe_image_positions(left, right, top, bottom, image_width, image_height):
    """
    Generate safe bubble positions relative to the actual image content
    """
    positions = []
    
    # Calculate margins to keep bubbles within image bounds
    margin_x = BUBBLE_WIDTH / 2 + 20
    margin_y = BUBBLE_HEIGHT / 2 + 20
    
    # Define grid within the image area
    grid_cols = 4
    grid_rows = 3
    
    # Calculate grid cell size within image
    cell_width = image_width / grid_cols
    cell_height = image_height / grid_rows
    
    # Generate grid positions within image
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = left + col * cell_width + cell_width / 2
            y = top + row * cell_height + cell_height / 2
            
            # Ensure bubble fits within image bounds
            if (left + margin_x <= x <= right - margin_x and 
                top + margin_y <= y <= bottom - margin_y):
                positions.append((x, y))
    
    # Add corner positions relative to image
    corner_margin = 30
    corners = [
        (left + corner_margin, top + corner_margin),  # Top-left of image
        (right - corner_margin, top + corner_margin),  # Top-right of image
        (left + corner_margin, bottom - corner_margin),  # Bottom-left of image
        (right - corner_margin, bottom - corner_margin)  # Bottom-right of image
    ]
    
    for corner in corners:
        if (left + margin_x <= corner[0] <= right - margin_x and 
            top + margin_y <= corner[1] <= bottom - margin_y):
            positions.append(corner)
    
    # Add edge positions along image boundaries
    edge_positions = []
    edge_margin = 50
    
    # Top edge of image
    for i in range(1, grid_cols):
        x = left + i * cell_width
        y = top + edge_margin
        if (left + margin_x <= x <= right - margin_x and 
            top + margin_y <= y <= bottom - margin_y):
            edge_positions.append((x, y))
    
    # Right edge of image
    for i in range(1, grid_rows):
        x = right - edge_margin
        y = top + i * cell_height
        if (left + margin_x <= x <= right - margin_x and 
            top + margin_y <= y <= bottom - margin_y):
            edge_positions.append((x, y))
    
    positions.extend(edge_positions)
    
    # If still no positions, use image center
    if len(positions) == 0:
        center_x = left + image_width / 2
        center_y = top + image_height / 2
        positions.append((center_x, center_y))
        print(f"Warning: Image too small, using center position only")
    
    print(f"Generated {len(positions)} safe positions for image area {image_width:.0f}x{image_height:.0f}")
    return positions

def _select_best_image_position(positions, left, right, top, bottom):
    """
    Select the best position relative to image content
    Priority: corners > edges > center
    """
    if not positions:
        # Fallback to image center if no positions available
        return (left + (right - left) / 2, top + (bottom - top) / 2)
    
    # Score positions based on preference
    scored_positions = []
    for pos in positions:
        x, y = pos
        score = 0
        
        # Prefer corners of image (highest score)
        corner_threshold = 50
        if (x < left + corner_threshold or x > right - corner_threshold) and \
           (y < top + corner_threshold or y > bottom - corner_threshold):
            score += 100
        
        # Prefer edges of image (medium score)
        edge_threshold = 80
        if (x < left + edge_threshold or x > right - edge_threshold) or \
           (y < top + edge_threshold or y > bottom - edge_threshold):
            score += 50
        
        # Prefer top and right areas (common comic bubble placement)
        if y < top + (bottom - top) / 2:  # Top half of image
            score += 25
        if x > left + (right - left) / 2:  # Right half of image
            score += 25
        
        scored_positions.append((pos, score))
    
    # Sort by score (highest first) and return the best
    scored_positions.sort(key=lambda x: x[1], reverse=True)
    best_position = scored_positions[0][0]
    
    print(f"Selected position with score {scored_positions[0][1]}")
    return best_position