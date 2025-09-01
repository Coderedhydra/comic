import os
from os import listdir
from backend.panel_layout.cam import get_coordinates, dump_CAM_data
from backend.utils import crop_image
from backend.panel_layout.layout.page import get_templates,panel_create
from backend.utils import get_panel_type, types
from PIL import Image


def smart_resize(index, panel_type, img_w, img_h):
    """
    Smart resize without cropping - maintains full image visibility and quality
    """
    frame_path = os.path.join("frames",'final',f"frame{index+1:03d}.png")
    wP, hP = types[panel_type]['width'], types[panel_type]['height']
    
    # Calculate scaling to fit image within panel while maintaining aspect ratio
    scale_w = wP / img_w
    scale_h = hP / img_h
    scale = min(scale_w, scale_h)  # Use smaller scale to fit entire image
    
    # Calculate new dimensions
    new_width = int(img_w * scale)
    new_height = int(img_h * scale)
    
    # Calculate centering offsets
    offset_x = (wP - new_width) / 2
    offset_y = (hP - new_height) / 2
    
    # Resize image with maximum quality settings
    img = Image.open(frame_path)
    
    # Use high-quality resampling for better image quality
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Save with maximum quality settings
    resized_img.save(frame_path, quality=95, optimize=True)
    
    # Return coordinates for bubble positioning (full image area)
    return (offset_x, offset_x + new_width, offset_y, offset_y + new_height)


def generate_layout():
    """
    Redesigned layout generation - no cropping, smart resizing, proper bubble alignment
    """
    # Get dimensions of images
    img = Image.open(os.path.join("frames",'final',f"frame001.png"))
    width, height = img.size
    
    # For high-accuracy mode, use simple panel type assignment
    HIGH_ACCURACY = os.getenv('HIGH_ACCURACY', '0')
    if HIGH_ACCURACY in ('1', 'true', 'True', 'YES', 'yes'):
        # Use panel type 6 (2x2) for all images in high-accuracy mode
        input_seq = ""
        folder_dir = "frames/final"
        for image in os.listdir(folder_dir):
            input_seq += "6"  # Always use panel type 6 for 2x2 grid
    else:
        # Original logic for non-high-accuracy mode
        input_seq = ""
        cam_coords = []
        folder_dir = "frames/final"
        for image in os.listdir(folder_dir):
            frame_path = os.path.join("frames",'final',image)
            left, right, top, bottom = get_coordinates(frame_path)
            input_seq += get_panel_type(left, right, top, bottom)
            cam_coords.append((left, right, top, bottom))
    
    page_templates = get_templates(input_seq)
    print(f"Page templates: {page_templates}")
    
    i = 0
    image_coords = []
    try:
        for page in page_templates:
            for panel in page:
                # Use smart resize instead of cropping
                coords = smart_resize(i, panel, width, height)
                image_coords.append(coords)
                i += 1
    except(IndexError):
        pass

    panels = panel_create(page_templates)
    
    # For high-accuracy mode, skip CAM data (not needed for smart resize)
    if HIGH_ACCURACY not in ('1', 'true', 'True', 'YES', 'yes'):
        dump_CAM_data()
    
    return image_coords, page_templates, panels