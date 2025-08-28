from backend.utils import convert_to_css_pixel, get_panel_type, types

BUBBLE_WIDTH = 200
BUUBLE_HEIGHT = 94

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
    left, right, top, bottom = crop_coord
    x_ = CAM_data['x_']
    y_ = CAM_data['y_']
    ten_map = CAM_data['ten_map']
    print("CAM map shape:", ten_map.shape)

    new_top = int(top / y_)
    new_bottom = int(bottom / y_)
    new_left = int(left / x_)
    new_right = int(right / x_)
    print("Panel bounds in CAM coords:", new_top, new_bottom, new_left, new_right)

    # Create face exclusion zones if lip coordinates are provided
    face_exclusion_zones = []
    if lip_coords and lip_coords[0] != -1 and lip_coords[1] != -1:
        lip_x, lip_y = lip_coords
        # Convert lip coords to CAM coordinate system
        cam_lip_x = int((lip_x + left) / x_)
        cam_lip_y = int((lip_y + top) / y_)
        # Create exclusion zone around lip (avoid 3x3 area around lip)
        exclusion_radius = 2
        for dx in range(-exclusion_radius, exclusion_radius + 1):
            for dy in range(-exclusion_radius, exclusion_radius + 1):
                ex_x = cam_lip_x + dx
                ex_y = cam_lip_y + dy
                if (new_left <= ex_x <= new_right and new_top <= ex_y <= new_bottom):
                    face_exclusion_zones.append((ex_x, ex_y))
        print(f"Created {len(face_exclusion_zones)} face exclusion zones around lip at ({cam_lip_x}, {cam_lip_y})")

    # Find areas with LOWEST activation (avoid faces, prefer background)
    # We want to place bubbles in areas with minimal CAM activation
    min_value = float('inf')
    min_point = None
    valid_candidates = []

    for i in range(new_left, new_right + 1):
        for j in range(new_top, new_bottom + 1):
            if (i < ten_map.shape[0] and j < ten_map.shape[1]):
                # Skip face exclusion zones
                if (i, j) in face_exclusion_zones:
                    continue
                # Prefer areas with very low CAM activation (background areas)
                activation = ten_map[i][j]
                if activation < 0.1:  # Very low activation threshold
                    valid_candidates.append((i, j, activation))
                elif activation < min_value:
                    min_value = activation
                    min_point = (i, j)

    # If we found good background candidates, use the one with lowest activation
    if valid_candidates:
        valid_candidates.sort(key=lambda x: x[2])  # Sort by activation value
        min_point = (valid_candidates[0][0], valid_candidates[0][1])
        print(f"Selected background candidate with activation {valid_candidates[0][2]}")
    elif min_point is None:
        # Fallback: use corner positions if no good candidates found
        min_point = (new_left + 2, new_top + 2)  # Small offset from corner
        print("Using fallback corner position")

    least_roi_x = min_point[0] * x_
    least_roi_y = min_point[1] * y_

    if least_roi_x < left:
        least_roi_x = left
    elif least_roi_x > right:
        least_roi_x = right
    if least_roi_y < top:
        least_roi_y = top
    elif least_roi_y > bottom:
        least_roi_y = bottom

    least_roi_x -= left
    least_roi_y -= top
    print("Selected position in image coords: ", least_roi_x, least_roi_y)
    
    least_roi_x, least_roi_y = convert_to_css_pixel(least_roi_x,least_roi_y,crop_coord)
    print("Position after CSS scaling: ", least_roi_x, least_roi_y)

    least_roi_x, least_roi_y = add_bubble_padding(least_roi_x, least_roi_y, crop_coord)

    return least_roi_x, least_roi_y