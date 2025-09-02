"""
Generate image files for each comic page at 800x1080 resolution using OpenCV
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple
import json

class PageImageGenerator:
    """Generate individual page images from comic data using OpenCV"""
    
    def __init__(self, output_dir: str = "output/page_images"):
        self.output_dir = output_dir
        self.page_size = (800, 1080)  # Width x Height
        self.panel_border = 3
        self.panel_gap = 10
        self.page_padding = 20
        
    def generate_page_images(self, pages_data: List[Dict], frames_dir: str) -> List[str]:
        """Generate images for all comic pages"""
        os.makedirs(self.output_dir, exist_ok=True)
        generated_files = []
        
        for i, page in enumerate(pages_data):
            page_image = self._create_page_image(page, frames_dir, i + 1)
            filename = f"page_{i+1:03d}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, page_image)
            generated_files.append(filepath)
            print(f"📄 Generated page {i+1}/{len(pages_data)}: {filename}")
            
        # Create an index file
        self._create_index_html(len(pages_data))
        
        return generated_files
    
    def _create_page_image(self, page: Dict, frames_dir: str, page_num: int) -> np.ndarray:
        """Create a single page image with 2x2 panel grid"""
        # Create white background
        page_img = np.ones((self.page_size[1], self.page_size[0], 3), dtype=np.uint8) * 255
        
        # Calculate panel dimensions (2x2 grid)
        available_width = self.page_size[0] - (2 * self.page_padding) - self.panel_gap
        available_height = self.page_size[1] - (2 * self.page_padding) - self.panel_gap
        
        panel_width = available_width // 2
        panel_height = available_height // 2
        
        # Process each panel in the 2x2 grid
        panels = page.get('panels', [])
        for idx, panel in enumerate(panels[:4]):  # Max 4 panels per page
            if not panel.get('image'):
                continue
                
            # Calculate position in grid (row, col)
            row = idx // 2
            col = idx % 2
            
            # Calculate panel position
            x = self.page_padding + col * (panel_width + self.panel_gap)
            y = self.page_padding + row * (panel_height + self.panel_gap)
            
            # Load and resize panel image
            img_path = os.path.join(frames_dir, panel['image'])
            if os.path.exists(img_path):
                panel_img = cv2.imread(img_path)
                if panel_img is not None:
                    # Calculate aspect ratio preserving resize
                    h, w = panel_img.shape[:2]
                    target_w = panel_width - 2*self.panel_border
                    target_h = panel_height - 2*self.panel_border
                    
                    # Calculate scale to fit within target dimensions
                    scale = min(target_w/w, target_h/h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # Resize image
                    panel_img = cv2.resize(panel_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    
                    # Center image in panel
                    img_x = x + (panel_width - new_w) // 2
                    img_y = y + (panel_height - new_h) // 2
                    
                    # Draw panel border
                    cv2.rectangle(page_img, (x, y), (x + panel_width, y + panel_height), 
                                 (0, 0, 0), self.panel_border)
                    
                    # Paste panel image
                    page_img[img_y:img_y+new_h, img_x:img_x+new_w] = panel_img
                    
                    # Add speech bubble if present
                    if panel.get('speech_bubble'):
                        self._add_speech_bubble(page_img, panel['speech_bubble'], 
                                              (x, y, panel_width, panel_height))
        
        # Add page number
        self._add_page_number(page_img, page_num)
        
        return page_img
    
    def _add_speech_bubble(self, img: np.ndarray, bubble_data: Dict, 
                          panel_bounds: Tuple[int, int, int, int]):
        """Add speech bubble to panel"""
        x, y, width, height = panel_bounds
        
        # Calculate bubble position relative to panel
        bubble_x = x + int(width * bubble_data.get('x', 0.5))
        bubble_y = y + int(height * bubble_data.get('y', 0.2))
        
        # Bubble dimensions
        bubble_width = min(int(width * 0.6), 200)
        bubble_height = 60
        
        # Draw bubble background (rounded rectangle approximation)
        bubble_rect = [
            bubble_x - bubble_width//2, 
            bubble_y - bubble_height//2,
            bubble_x + bubble_width//2, 
            bubble_y + bubble_height//2
        ]
        
        # Draw white filled rectangle
        cv2.rectangle(img, 
                     (bubble_rect[0], bubble_rect[1]),
                     (bubble_rect[2], bubble_rect[3]),
                     (255, 255, 255), -1)
        
        # Draw black border
        cv2.rectangle(img, 
                     (bubble_rect[0], bubble_rect[1]),
                     (bubble_rect[2], bubble_rect[3]),
                     (0, 0, 0), 2)
        
        # Add text
        text = bubble_data.get('text', '')
        if text:
            # Simple text wrapping
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 20:  # Simple wrap at 20 chars
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw each line
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            line_height = 20
            
            for i, line in enumerate(lines[:3]):  # Max 3 lines
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                text_x = bubble_x - text_size[0] // 2
                text_y = bubble_y - (len(lines) * line_height) // 2 + (i + 1) * line_height
                
                cv2.putText(img, line, (text_x, text_y), font, 
                           font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    def _add_page_number(self, img: np.ndarray, page_num: int):
        """Add page number at bottom"""
        text = f"Page {page_num}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (self.page_size[0] - text_size[0]) // 2
        y = self.page_size[1] - 15
        
        cv2.putText(img, text, (x, y), font, font_scale, 
                   (128, 128, 128), thickness, cv2.LINE_AA)
    
    def _create_index_html(self, num_pages: int):
        """Create an HTML index for viewing page images"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Comic Page Images</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f0f0f0;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .page-card {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .page-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .page-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .page-info {
            padding: 10px;
            text-align: center;
            font-size: 14px;
            color: #666;
        }
        .download-all {
            display: inline-block;
            margin: 20px auto;
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        }
        .download-all:hover {
            background: #45a049;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 Comic Page Images</h1>
        <p>All pages rendered at 800x1080 resolution</p>
        <a href="#" class="download-all" onclick="downloadAll()">⬇️ Download All Pages</a>
    </div>
    
    <div class="gallery">
"""
        
        # Add each page
        for i in range(num_pages):
            page_num = i + 1
            filename = f"page_{page_num:03d}.png"
            html_content += f"""
        <div class="page-card">
            <a href="{filename}" download>
                <img src="{filename}" alt="Page {page_num}">
            </a>
            <div class="page-info">
                Page {page_num} 
                <a href="{filename}" download>⬇️ Download</a>
            </div>
        </div>
"""
        
        html_content += """
    </div>
    
    <script>
        function downloadAll() {
            const links = document.querySelectorAll('.page-card a[download]');
            links.forEach((link, index) => {
                setTimeout(() => {
                    link.click();
                }, index * 200);  // Stagger downloads
            });
        }
    </script>
</body>
</html>
"""
        
        index_path = os.path.join(self.output_dir, 'index.html')
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        print(f"📋 Page index created: {index_path}")

def generate_page_images_from_json(json_path: str, frames_dir: str, output_dir: str = None):
    """Standalone function to generate page images from pages.json"""
    if not os.path.exists(json_path):
        print(f"❌ Pages JSON not found: {json_path}")
        return []
    
    # Load pages data
    with open(json_path, 'r') as f:
        pages_data = json.load(f)
    
    # Create generator
    generator = PageImageGenerator(output_dir or "output/page_images")
    
    # Generate images
    return generator.generate_page_images(pages_data, frames_dir)