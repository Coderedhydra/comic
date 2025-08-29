import math
import json
import srt
import pickle
import os
from backend.speech_bubble.lip_detection import get_lips
from backend.speech_bubble.bubble_placement import get_bubble_position
from backend.speech_bubble.bubble_shape import get_bubble_type
from backend.class_def import bubble
import threading
from backend.utils import get_panel_type, types


def _does_overlap(new_bubble, existing_bubbles, bubble_width=200, bubble_height=94, padding=8):
    nx1 = new_bubble[0]
    ny1 = new_bubble[1]
    nx2 = nx1 + bubble_width + padding
    ny2 = ny1 + bubble_height + padding
    for (ex, ey) in existing_bubbles:
        ex1 = ex
        ey1 = ey
        ex2 = ex1 + bubble_width + padding
        ey2 = ey1 + bubble_height + padding
        if not (nx2 <= ex1 or ex2 <= nx1 or ny2 <= ey1 or ey2 <= ny1):
            return True
    return False

def _clamp_to_panel(px, py, crop_coord, bubble_width=200, bubble_height=94):
    left, right, top, bottom = crop_coord
    panel = get_panel_type(left, right, top, bottom)
    panel_w = types[panel]['width']
    panel_h = types[panel]['height']
    px = max(0, min(px, panel_w - bubble_width))
    py = max(0, min(py, panel_h - bubble_height))
    return px, py

def _avoid_lip_overlap(px, py, lip_x, lip_y, crop_coord, bubble_width=200, bubble_height=94):
    if lip_x == -1 and lip_y == -1:
        return px, py
    
    # Create a larger exclusion zone around the lip (face area)
    face_margin = 60  # Increased margin around face
    rect_x1, rect_y1 = px, py
    rect_x2, rect_y2 = px + bubble_width, py + bubble_height
    
    # Check if bubble overlaps with face exclusion zone
    face_x1 = lip_x - face_margin
    face_y1 = lip_y - face_margin  
    face_x2 = lip_x + face_margin
    face_y2 = lip_y + face_margin
    
    # If bubble overlaps face zone, push it away
    if not (rect_x2 <= face_x1 or face_x2 <= rect_x1 or rect_y2 <= face_y1 or face_y2 <= rect_y1):
        # Calculate push direction: away from face center
        bubble_center_x = (rect_x1 + rect_x2) / 2.0
        bubble_center_y = (rect_y1 + rect_y2) / 2.0
        
        # Vector from face to bubble center
        vx = bubble_center_x - lip_x
        vy = bubble_center_y - lip_y
        
        # Normalize and push
        if vx == 0 and vy == 0:
            vx, vy = 1.0, 0.0  # Default push right if same position
        
        mag = (vx**2 + vy**2) ** 0.5
        ux, uy = vx / mag, vy / mag
        
        # Push bubble away from face
        push_distance = face_margin + max(bubble_width, bubble_height) / 2
        px += ux * push_distance
        py += uy * push_distance
        
        # Ensure bubble stays within panel bounds
        px, py = _clamp_to_panel(px, py, crop_coord, bubble_width, bubble_height)
        
        print(f"Pushed bubble away from face at ({lip_x}, {lip_y}) to ({px}, {py})")
    
    return px, py

def bubble_create(video, crop_coords, black_x, black_y):

    bubbles = []


    # def bubble_create(bubble_cord,lip_cord,page_template):
    data=""
    with open("test1.srt") as f:
        data=f.read()
    subs=srt.parse(data)


    # Reading CAM data from dump (only for legacy mode)
    HIGH_ACCURACY = os.getenv('HIGH_ACCURACY', '0')
    CAM_data = None
    if HIGH_ACCURACY not in ('1', 'true', 'True', 'YES', 'yes'):
        try:
            with open('CAM_data.pkl', 'rb') as f:
                CAM_data = pickle.load(f)
        except FileNotFoundError:
            print("Warning: CAM_data.pkl not found, using high-accuracy mode")
            CAM_data = None

    lips = get_lips(video, crop_coords,black_x,black_y)
    # Dumping lips
    with open('lips.pkl', 'wb') as f:
        pickle.dump(lips, f)

    # # Reading lips
    # lips=None
    # with open('lips.pkl', 'rb') as f:
    #     lips = pickle.load(f)
    
    # emotion_thread.join()
    # print("Detected emotions:", emotions)


    placed_positions = []
    for sub in subs:
        lip_x = lips[sub.index][0]
        lip_y = lips[sub.index][1]

        # Use smart bubble positioning system
        HIGH_ACCURACY = os.getenv('HIGH_ACCURACY', '0')
        if HIGH_ACCURACY in ('1', 'true', 'True', 'YES', 'yes'):
            # Use smart image analysis for bubble placement
            try:
                from backend.speech_bubble.smart_bubble_placement import get_smart_bubble_position
                frame_path = f"frames/final/frame{sub.index:03}.png"
                bubble_x, bubble_y = get_smart_bubble_position(frame_path, crop_coords[sub.index-1], (lip_x, lip_y))
                print(f"Smart placement: ({bubble_x:.0f}, {bubble_y:.0f})")
            except Exception as e:
                print(f"Smart placement failed: {e}, using fallback")
                # Fallback to simple upper positioning
                left, right, top, bottom = crop_coords[sub.index-1]
                bubble_x = left + (right - left) * 0.8  # 80% from left
                bubble_y = top + (bottom - top) * 0.2   # 20% from top
        else:
            # For legacy mode, use CAM data
            bubble_x, bubble_y = get_bubble_position(crop_coords[sub.index-1], CAM_data[sub.index-1], (lip_x, lip_y))

        # Advanced collision avoidance with grid-based positioning
        px, py = bubble_x, bubble_y
        
        # First, try to avoid face overlap
        px, py = _avoid_lip_overlap(px, py, lip_x, lip_y, crop_coords[sub.index-1])
        
        # Then handle bubble-to-bubble collision with smart positioning
        attempts = 0
        max_attempts = 15
        original_pos = (px, py)
        
        while _does_overlap((px, py), placed_positions) and attempts < max_attempts:
            # Try different directions in order of preference
            directions = [
                (40, 0),   # Right
                (0, -40),  # Up
                (-40, 0),  # Left
                (0, 40),   # Down
                (40, -40), # Up-right
                (-40, -40), # Up-left
                (40, 40),  # Down-right
                (-40, 40), # Down-left
            ]
            
            if attempts < len(directions):
                dx, dy = directions[attempts]
                px = original_pos[0] + dx
                py = original_pos[1] + dy
            else:
                # Spiral outward if all directions fail
                angle = attempts * 0.5
                radius = 20 + attempts * 10
                px = original_pos[0] + radius * math.cos(angle)
                py = original_pos[1] + radius * math.sin(angle)
            
            # Ensure position stays within panel bounds
            px, py = _clamp_to_panel(px, py, crop_coords[sub.index-1])
            attempts += 1
        
        bubble_x, bubble_y = px, py
        placed_positions.append((bubble_x, bubble_y))

        dialogue = sub.content
        emotion = get_bubble_type(dialogue)
        print(f'||emotion:{emotion}||')


        temp = bubble(bubble_x, bubble_y,lip_x,lip_y,sub.content,emotion)
        bubbles.append(temp)

    return bubbles









