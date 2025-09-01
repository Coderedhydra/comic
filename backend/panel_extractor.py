"""
Panel Extractor - Extracts and saves individual comic panels as 640x800 images
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple

class PanelExtractor:
    def __init__(self, output_dir: str = "output/panels"):
        """Initialize panel extractor
        
        Args:
            output_dir: Directory to save extracted panels
        """
        self.output_dir = output_dir
        self.panel_size = (640, 800)  # Width x Height
        
    def extract_panels_from_comic(self, pages_json_path: str = "output/pages.json", 
                                 frames_dir: str = "frames/final") -> List[str]:
        """Extract panels from generated comic data
        
        Args:
            pages_json_path: Path to pages.json file
            frames_dir: Directory containing frame images
            
        Returns:
            List of saved panel file paths
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Clear existing panels
        for file in os.listdir(self.output_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                os.remove(os.path.join(self.output_dir, file))
        
        # Load comic data
        try:
            with open(pages_json_path, 'r') as f:
                pages_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load comic data: {e}")
            return []
            
        saved_panels = []
        panel_count = 0
        
        print(f"📸 Extracting panels as {self.panel_size[0]}x{self.panel_size[1]} images...")
        
        # Process each page
        for page_idx, page in enumerate(pages_data):
            panels = page.get('panels', [])
            bubbles = page.get('bubbles', [])
            
            # Process each panel
            for panel_idx, panel in enumerate(panels):
                panel_count += 1
                
                # Extract panel image
                panel_img = self._extract_panel(panel, frames_dir)
                if panel_img is None:
                    continue
                    
                # Find bubbles that belong to this panel
                panel_bubbles = self._find_panel_bubbles(panel, bubbles)
                
                # Add bubbles to panel
                if panel_bubbles:
                    panel_img = self._add_bubbles_to_panel(panel_img, panel, panel_bubbles)
                
                # Resize to target size
                panel_img = self._resize_panel(panel_img)
                
                # Save panel
                filename = f"panel_{panel_count:03d}_p{page_idx+1}_{panel_idx+1}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                # Convert to RGB if needed (remove alpha channel)
                if len(panel_img.shape) == 3 and panel_img.shape[2] == 4:
                    panel_img = cv2.cvtColor(panel_img, cv2.COLOR_BGRA2BGR)
                elif len(panel_img.shape) == 2:
                    panel_img = cv2.cvtColor(panel_img, cv2.COLOR_GRAY2BGR)
                
                cv2.imwrite(filepath, panel_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_panels.append(filepath)
                
        print(f"✅ Extracted {len(saved_panels)} panels to: {self.output_dir}")
        
        # Create an index HTML for viewing
        self._create_panel_viewer(saved_panels)
        
        return saved_panels
    
    def _extract_panel(self, panel: Dict, frames_dir: str) -> np.ndarray:
        """Extract panel region from frame image"""
        try:
            # Get frame path
            frame_filename = os.path.basename(panel['image'])
            frame_path = os.path.join(frames_dir, frame_filename)
            
            if not os.path.exists(frame_path):
                # Try without 'final' in path
                frame_path = panel['image'].lstrip('/')
                if not os.path.exists(frame_path):
                    print(f"⚠️ Frame not found: {frame_path}")
                    return None
            
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"⚠️ Failed to load frame: {frame_path}")
                return None
            
            # No need to extract region - use full frame as is
            # The panel coordinates are for HTML display, not image cropping
            return frame
            
        except Exception as e:
            print(f"❌ Failed to extract panel: {e}")
            return None
    
    def _find_panel_bubbles(self, panel: Dict, bubbles: List[Dict]) -> List[Dict]:
        """Find speech bubbles that belong to a panel"""
        panel_bubbles = []
        
        # Panel boundaries
        px1 = panel['x']
        py1 = panel['y']
        px2 = px1 + panel['width']
        py2 = py1 + panel['height']
        
        for bubble in bubbles:
            # Bubble center
            bx = bubble['x'] + bubble['width'] / 2
            by = bubble['y'] + bubble['height'] / 2
            
            # Check if bubble center is within panel
            if px1 <= bx <= px2 and py1 <= by <= py2:
                # Adjust bubble coordinates relative to panel
                adjusted_bubble = bubble.copy()
                adjusted_bubble['x'] -= px1
                adjusted_bubble['y'] -= py1
                panel_bubbles.append(adjusted_bubble)
                
        return panel_bubbles
    
    def _add_bubbles_to_panel(self, panel_img: np.ndarray, panel: Dict, 
                             bubbles: List[Dict]) -> np.ndarray:
        """Add speech bubbles to panel image"""
        # Convert to PIL for easier drawing
        img = Image.fromarray(cv2.cvtColor(panel_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        
        # Try to load a comic font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
        except:
            font = None
        
        for bubble in bubbles:
            # Scale bubble coordinates to match image size
            img_h, img_w = panel_img.shape[:2]
            panel_w, panel_h = panel['width'], panel['height']
            
            # Scale factors
            scale_x = img_w / panel_w
            scale_y = img_h / panel_h
            
            # Scaled coordinates
            x = int(bubble['x'] * scale_x)
            y = int(bubble['y'] * scale_y)
            w = int(bubble['width'] * scale_x)
            h = int(bubble['height'] * scale_y)
            
            # Draw bubble background
            bubble_bbox = [x, y, x + w, y + h]
            draw.ellipse(bubble_bbox, fill='white', outline='black', width=2)
            
            # Draw text
            text = bubble.get('text', '')
            if text and font:
                # Word wrap text
                words = text.split()
                lines = []
                current_line = []
                
                for word in words:
                    current_line.append(word)
                    line_text = ' '.join(current_line)
                    bbox = draw.textbbox((0, 0), line_text, font=font)
                    if bbox[2] > w - 20:  # Leave padding
                        if len(current_line) > 1:
                            current_line.pop()
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            lines.append(line_text)
                            current_line = []
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Draw centered text
                line_height = 20
                total_height = len(lines) * line_height
                start_y = y + (h - total_height) // 2
                
                for i, line in enumerate(lines):
                    bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_x = x + (w - text_width) // 2
                    text_y = start_y + i * line_height
                    draw.text((text_x, text_y), line, fill='black', font=font)
        
        # Convert back to numpy
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    def _resize_panel(self, panel_img: np.ndarray) -> np.ndarray:
        """Resize panel to target size (640x800)"""
        h, w = panel_img.shape[:2]
        target_w, target_h = self.panel_size
        
        # Calculate scale to fit within target size while maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(panel_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create canvas of target size
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255  # White background
        
        # Center the resized image on canvas
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def _create_panel_viewer(self, panel_files: List[str]):
        """Create an HTML viewer for extracted panels"""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Extracted Comic Panels - 640x800</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: white;
            font-family: Arial, sans-serif;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
        }
        .panel-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .panel-card {
            background: #2a2a2a;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.3s;
        }
        .panel-card:hover {
            transform: scale(1.05);
        }
        .panel-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .panel-info {
            padding: 10px;
            text-align: center;
            font-size: 14px;
            color: #aaa;
        }
        .download-all {
            display: block;
            margin: 20px auto;
            padding: 10px 30px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            max-width: 200px;
        }
        .download-all:hover {
            background: #45a049;
        }
    </style>
</head>
<body>
    <h1>📸 Extracted Comic Panels (640x800)</h1>
    <p style="text-align: center; color: #888;">All panels have been extracted and resized to 640x800 pixels</p>
    
    <div class="panel-grid">
'''
        
        for panel_path in panel_files:
            filename = os.path.basename(panel_path)
            panel_num = filename.split('_')[1]
            
            html += f'''
        <div class="panel-card">
            <img src="{filename}" alt="{filename}">
            <div class="panel-info">Panel {panel_num}</div>
        </div>
'''
        
        html += '''
    </div>
</body>
</html>'''
        
        viewer_path = os.path.join(self.output_dir, 'panel_viewer.html')
        with open(viewer_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
        print(f"📄 Panel viewer created: {viewer_path}")


# Convenience function for command line usage
def extract_panels(pages_json: str = "output/pages.json", 
                  frames_dir: str = "frames/final",
                  output_dir: str = "output/panels"):
    """Extract panels from comic"""
    extractor = PanelExtractor(output_dir)
    return extractor.extract_panels_from_comic(pages_json, frames_dir)


if __name__ == "__main__":
    extract_panels()