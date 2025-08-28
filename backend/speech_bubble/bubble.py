import math
import json
import srt
import pickle
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
    # If the lip lies inside the bubble rectangle, push the bubble away
    rect_x1, rect_y1 = px, py
    rect_x2, rect_y2 = px + bubble_width, py + bubble_height
    if rect_x1 <= lip_x <= rect_x2 and rect_y1 <= lip_y <= rect_y2:
        # Push along vector from lip to bubble center
        cx = (rect_x1 + rect_x2) / 2.0
        cy = (rect_y1 + rect_y2) / 2.0
        vx = cx - lip_x
        vy = cy - lip_y
        if vx == 0 and vy == 0:
            vx, vy = 1.0, 0.0
        mag = (vx**2 + vy**2) ** 0.5
        ux, uy = vx / mag, vy / mag
        step = 24
        max_steps = 20
        for _ in range(max_steps):
            px += ux * step
            py += uy * step
            px, py = _clamp_to_panel(px, py, crop_coord, bubble_width, bubble_height)
            rect_x1, rect_y1 = px, py
            rect_x2, rect_y2 = px + bubble_width, py + bubble_height
            if not (rect_x1 <= lip_x <= rect_x2 and rect_y1 <= lip_y <= rect_y2):
                break
    return px, py

def bubble_create(video, crop_coords, black_x, black_y):

    bubbles = []


    # def bubble_create(bubble_cord,lip_cord,page_template):
    data=""
    with open("test1.srt") as f:
        data=f.read()
    subs=srt.parse(data)


    # Reading CAM data from dump
    CAM_data = None
    with open('CAM_data.pkl', 'rb') as f:
        CAM_data = pickle.load(f)

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

        bubble_x, bubble_y = get_bubble_position(crop_coords[sub.index-1], CAM_data[sub.index-1])

        # Simple collision avoidance: nudge right/down in steps until no overlap or max attempts
        max_attempts = 10
        step = 16
        attempts = 0
        px, py = bubble_x, bubble_y
        while _does_overlap((px, py), placed_positions) and attempts < max_attempts:
            px += step
            py += step // 2
            attempts += 1
        # Avoid covering the detected lips/face area
        px, py = _avoid_lip_overlap(px, py, lip_x, lip_y, crop_coords[sub.index-1])
        bubble_x, bubble_y = _clamp_to_panel(px, py, crop_coords[sub.index-1])
        placed_positions.append((bubble_x, bubble_y))

        dialogue = sub.content
        emotion = get_bubble_type(dialogue)
        print(f'||emotion:{emotion}||')


        temp = bubble(bubble_x, bubble_y,lip_x,lip_y,sub.content,emotion)
        bubbles.append(temp)

    return bubbles









