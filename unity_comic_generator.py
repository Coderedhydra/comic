"""
Unity Comic Generator
Creates 48 pages with 2x2 grid layout for Unity integration
High-quality PNG output with interactive bubble editing
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import srt
from typing import List, Dict, Tuple
import math

class UnityComicGenerator:
    def __init__(self):
        self.panel_width = 800
        self.panel_height = 540
        self.page_width = 1600  # 2 panels * 800
        self.page_height = 1080  # 2 panels * 540
        self.total_pages = 48
        self.panels_per_page = 4  # 2x2 grid
        
        # Create output directories
        os.makedirs('output/unity_pages', exist_ok=True)
        os.makedirs('output/unity_panels', exist_ok=True)
        os.makedirs('output/unity_bubbles', exist_ok=True)
        
    def generate_48_pages_comic(self, video_path: str = 'video/uploaded.mp4'):
        """Generate 48 pages comic with 2x2 grid layout"""
        print("🎬 Starting Unity Comic Generation (48 pages, 2x2 grid)...")
        
        try:
            # 1. Extract frames
            frames = self._extract_frames(video_path)
            print(f"📸 Extracted {len(frames)} frames")
            
            # 2. Resize frames to 800x540
            resized_frames = self._resize_frames_for_unity(frames)
            print(f"📐 Resized {len(resized_frames)} frames to 800x540")
            
            # 3. Extract subtitles
            subtitles = self._extract_subtitles()
            print(f"💬 Extracted {len(subtitles)} subtitles")
            
            # 4. Generate 48 pages
            pages_data = self._generate_48_pages(resized_frames, subtitles)
            print(f"📄 Generated {len(pages_data)} pages")
            
            # 5. Create high-quality PNG pages
            png_pages = self._create_png_pages(pages_data)
            print(f"🖼️ Created {len(png_pages)} PNG pages")
            
            # 6. Generate interactive HTML viewer
            self._create_interactive_viewer(pages_data)
            print("🌐 Created interactive viewer")
            
            # 7. Save Unity-ready data
            self._save_unity_data(pages_data)
            print("💾 Saved Unity-ready data")
            
            print("✅ Unity Comic Generation Complete!")
            return True
            
        except Exception as e:
            print(f"❌ Unity Comic Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_frames(self, video_path: str) -> List[str]:
        """Extract frames from video"""
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return []
        
        frames_dir = "frames/unity_extracted"
        os.makedirs(frames_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted_frames = []
        
        # Calculate frame interval for 48 pages * 4 panels = 192 frames
        total_frames_needed = self.total_pages * self.panels_per_page
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_video_frames // total_frames_needed)
        
        print(f"📊 Video has {total_video_frames} frames, extracting every {frame_interval} frames")
        
        while frame_count < total_frames_needed:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{frame_count:03d}.png"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)
                
                if len(extracted_frames) >= total_frames_needed:
                    break
            
            frame_count += 1
        
        cap.release()
        return extracted_frames
    
    def _resize_frames_for_unity(self, frame_paths: List[str]) -> List[str]:
        """Resize frames to 800x540 for Unity"""
        resized_dir = "frames/unity_resized"
        os.makedirs(resized_dir, exist_ok=True)
        
        resized_frames = []
        
        for i, frame_path in enumerate(frame_paths):
            try:
                # Read original frame
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                
                # Get original dimensions
                h, w = img.shape[:2]
                
                # Calculate scale to fit within 800x540 without cropping
                scale = min(self.panel_width/w, self.panel_height/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize with high quality
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Create 800x540 canvas with black background
                canvas = np.zeros((self.panel_height, self.panel_width, 3), dtype=np.uint8)
                
                # Center the resized image
                y_offset = (self.panel_height - new_h) // 2
                x_offset = (self.panel_width - new_w) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                # Save resized frame
                output_path = os.path.join(resized_dir, f"resized_{i:03d}.png")
                cv2.imwrite(output_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # No compression for Unity
                resized_frames.append(output_path)
                
            except Exception as e:
                print(f"❌ Error resizing frame {frame_path}: {e}")
        
        return resized_frames
    
    def _extract_subtitles(self) -> List[Dict]:
        """Extract subtitles from SRT file"""
        subtitles = []
        
        if os.path.exists('test1.srt'):
            try:
                with open('test1.srt', 'r', encoding='utf-8') as f:
                    srt_content = f.read()
                    parsed_subs = list(srt.parse(srt_content))
                    
                    for sub in parsed_subs:
                        subtitles.append({
                            'text': sub.content,
                            'start': sub.start.total_seconds(),
                            'end': sub.end.total_seconds(),
                            'index': sub.index
                        })
            except Exception as e:
                print(f"⚠️ Error reading subtitles: {e}")
        
        # If no subtitles, create placeholder ones
        if not subtitles:
            for i in range(self.total_pages * self.panels_per_page):
                subtitles.append({
                    'text': f"Panel {i+1}",
                    'start': i * 2.0,
                    'end': (i + 1) * 2.0,
                    'index': i + 1
                })
        
        return subtitles
    
    def _generate_48_pages(self, frames: List[str], subtitles: List[Dict]) -> List[Dict]:
        """Generate 48 pages with 2x2 grid layout"""
        pages = []
        
        for page_num in range(self.total_pages):
            page_data = {
                'page_number': page_num + 1,
                'panels': [],
                'bubbles': []
            }
            
            # Create 4 panels for this page (2x2 grid)
            for panel_num in range(self.panels_per_page):
                frame_idx = page_num * self.panels_per_page + panel_num
                
                # Get frame (cycle if not enough frames)
                frame_path = frames[frame_idx % len(frames)] if frames else None
                
                # Calculate panel position in 2x2 grid
                row = panel_num // 2
                col = panel_num % 2
                
                panel_data = {
                    'panel_number': panel_num + 1,
                    'frame_path': frame_path,
                    'row': row,
                    'col': col,
                    'x': col * self.panel_width,
                    'y': row * self.panel_height,
                    'width': self.panel_width,
                    'height': self.panel_height
                }
                page_data['panels'].append(panel_data)
                
                # Create bubble for this panel
                subtitle_idx = frame_idx % len(subtitles)
                subtitle = subtitles[subtitle_idx]
                
                bubble_data = {
                    'panel_number': panel_num + 1,
                    'text': subtitle['text'],
                    'x': col * self.panel_width + 50,  # Offset from panel edge
                    'y': row * self.panel_height + 50,
                    'width': 200,
                    'height': 80,
                    'font_size': 16,
                    'color': '#000000',
                    'background': '#FFFFFF',
                    'border': '#000000'
                }
                page_data['bubbles'].append(bubble_data)
            
            pages.append(page_data)
        
        return pages
    
    def _create_png_pages(self, pages_data: List[Dict]) -> List[str]:
        """Create high-quality PNG pages for Unity"""
        png_pages = []
        
        for page_data in pages_data:
            try:
                # Create page canvas
                page_img = np.zeros((self.page_height, self.page_width, 3), dtype=np.uint8)
                page_img.fill(255)  # White background
                
                # Add panels to page
                for panel in page_data['panels']:
                    if panel['frame_path'] and os.path.exists(panel['frame_path']):
                        # Load panel image
                        panel_img = cv2.imread(panel['frame_path'])
                        if panel_img is not None:
                            # Place panel on page
                            y1 = panel['y']
                            y2 = y1 + panel['height']
                            x1 = panel['x']
                            x2 = x1 + panel['width']
                            
                            page_img[y1:y2, x1:x2] = panel_img
                
                # Add speech bubbles
                for bubble in page_data['bubbles']:
                    self._draw_bubble_on_image(page_img, bubble)
                
                # Save page as PNG
                page_filename = f"page_{page_data['page_number']:03d}.png"
                page_path = os.path.join('output/unity_pages', page_filename)
                cv2.imwrite(page_path, page_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                png_pages.append(page_path)
                
                print(f"✅ Created page {page_data['page_number']}: {page_filename}")
                
            except Exception as e:
                print(f"❌ Error creating page {page_data['page_number']}: {e}")
        
        return png_pages
    
    def _draw_bubble_on_image(self, img: np.ndarray, bubble: Dict):
        """Draw speech bubble on image"""
        try:
            # Convert to PIL for text rendering
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", bubble['font_size'])
            except:
                font = ImageFont.load_default()
            
            # Draw bubble background
            x, y = bubble['x'], bubble['y']
            w, h = bubble['width'], bubble['height']
            
            # Draw rounded rectangle background
            draw.rounded_rectangle([x, y, x+w, y+h], radius=10, 
                                 fill=bubble['background'], outline=bubble['border'], width=2)
            
            # Draw text
            text = bubble['text']
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center text in bubble
            text_x = x + (w - text_width) // 2
            text_y = y + (h - text_height) // 2
            
            draw.text((text_x, text_y), text, fill=bubble['color'], font=font)
            
            # Draw bubble tail
            tail_x = x + 20
            tail_y = y + h
            tail_points = [
                (tail_x, tail_y),
                (tail_x + 10, tail_y + 15),
                (tail_x - 10, tail_y + 15)
            ]
            draw.polygon(tail_points, fill=bubble['background'], outline=bubble['border'])
            
            # Convert back to OpenCV format
            img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"⚠️ Error drawing bubble: {e}")
    
    def _create_interactive_viewer(self, pages_data: List[Dict]):
        """Create interactive HTML viewer with draggable bubbles"""
        html_content = self._generate_interactive_html(pages_data)
        
        with open('output/unity_pages/interactive_viewer.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_interactive_html(self, pages_data: List[Dict]) -> str:
        """Generate interactive HTML with draggable bubbles"""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unity Comic Viewer - 48 Pages with Interactive Bubbles</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{ 
            background: #f0f0f0; 
            font-family: Arial, sans-serif; 
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: #333;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        
        .controls {{
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .btn {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }}
        
        .btn:hover {{ background: #45a049; }}
        .btn:disabled {{ background: #ccc; cursor: not-allowed; }}
        
        .page-container {{
            max-width: 100%;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-bottom: 20px;
        }}
        
        .comic-page {{
            width: 1600px;
            height: 1080px;
            margin: 0 auto;
            position: relative;
            background: white;
            border: 3px solid #333;
        }}
        
        .panel {{
            position: absolute;
            width: 800px;
            height: 540px;
            border: 1px solid #ddd;
        }}
        
        .panel img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}
        
        .speech-bubble {{
            position: absolute;
            background: white;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 8px 12px;
            max-width: 200px;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
            z-index: 10;
            text-align: center;
            color: #333;
            cursor: move;
            user-select: none;
            min-width: 100px;
            min-height: 40px;
        }}
        
        .speech-bubble:hover {{
            transform: scale(1.05);
            box-shadow: 2px 2px 10px rgba(0,0,0,0.4);
        }}
        
        .speech-bubble::after {{
            content: '';
            position: absolute;
            bottom: -8px;
            left: 15px;
            width: 0;
            height: 0;
            border-left: 8px solid transparent;
            border-right: 8px solid transparent;
            border-top: 8px solid #333;
        }}
        
        .speech-bubble.editing {{
            border-color: #ff6b6b;
            box-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
        }}
        
        .page-navigation {{
            text-align: center;
            margin: 20px 0;
        }}
        
        .page-info {{
            display: inline-block;
            margin: 0 20px;
            font-size: 18px;
            font-weight: bold;
        }}
        
        .loading {{
            text-align: center;
            padding: 50px;
            color: #666;
            font-size: 18px;
        }}
        
        .zoom-controls {{
            margin: 10px 0;
        }}
        
        .zoom-slider {{
            width: 200px;
            margin: 0 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Unity Comic Viewer</h1>
        <p>48 Pages • 2x2 Grid • Interactive Speech Bubbles</p>
        <p>Drag bubbles to move • Double-click to edit • Print individual pages</p>
    </div>
    
    <div class="controls">
        <button class="btn" onclick="previousPage()" id="prevBtn">⬅️ Previous</button>
        <span class="page-info">Page <span id="currentPage">1</span> of 48</span>
        <button class="btn" onclick="nextPage()" id="nextBtn">Next ➡️</button>
        
        <div class="zoom-controls">
            <label>Zoom: </label>
            <input type="range" class="zoom-slider" min="25" max="100" value="50" onchange="setZoom(this.value)">
            <span id="zoomValue">50%</span>
        </div>
        
        <button class="btn" onclick="printCurrentPage()">🖨️ Print This Page</button>
        <button class="btn" onclick="saveCurrentPage()">💾 Save PNG</button>
        <button class="btn" onclick="exportAllPages()">📦 Export All Pages</button>
    </div>
    
    <div class="page-container">
        <div id="comic-content">
            <div class="loading">Loading comic pages...</div>
        </div>
    </div>
    
    <script>
        let currentPageIndex = 0;
        let pagesData = {json.dumps(pages_data, indent=2)};
        let isDragging = false;
        let dragElement = null;
        let dragOffset = {{x: 0, y: 0}};
        let currentZoom = 0.5;
        
        // Initialize viewer
        document.addEventListener('DOMContentLoaded', function() {{
            loadPage(0);
        }});
        
        function loadPage(pageIndex) {{
            if (pageIndex < 0 || pageIndex >= pagesData.length) return;
            
            currentPageIndex = pageIndex;
            const pageData = pagesData[pageIndex];
            
            const container = document.getElementById('comic-content');
            container.innerHTML = '';
            
            // Create page
            const pageDiv = document.createElement('div');
            pageDiv.className = 'comic-page';
            pageDiv.style.transform = `scale(${{currentZoom}})`;
            pageDiv.style.transformOrigin = 'top center';
            
            // Add panels
            pageData.panels.forEach((panel, index) => {{
                const panelDiv = document.createElement('div');
                panelDiv.className = 'panel';
                panelDiv.style.left = panel.x + 'px';
                panelDiv.style.top = panel.y + 'px';
                
                const img = document.createElement('img');
                img.src = '/frames/unity_resized/' + panel.frame_path.split('/').pop();
                img.alt = `Panel ${{panel.panel_number}}`;
                
                // Handle missing images
                img.onerror = function() {{
                    this.style.display = 'none';
                    panelDiv.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; background: #ddd; color: #666; font-size: 18px;">Panel ' + panel.panel_number + '</div>';
                }};
                
                panelDiv.appendChild(img);
                pageDiv.appendChild(panelDiv);
            }});
            
            // Add speech bubbles
            pageData.bubbles.forEach((bubble, index) => {{
                const bubbleDiv = document.createElement('div');
                bubbleDiv.className = 'speech-bubble';
                bubbleDiv.textContent = bubble.text;
                bubbleDiv.style.left = bubble.x + 'px';
                bubbleDiv.style.top = bubble.y + 'px';
                bubbleDiv.style.width = bubble.width + 'px';
                bubbleDiv.style.height = bubble.height + 'px';
                bubbleDiv.style.fontSize = bubble.font_size + 'px';
                bubbleDiv.style.color = bubble.color;
                bubbleDiv.style.backgroundColor = bubble.background;
                bubbleDiv.style.borderColor = bubble.border;
                
                // Make bubble draggable
                bubbleDiv.addEventListener('mousedown', startDrag);
                bubbleDiv.addEventListener('dblclick', editBubble);
                
                pageDiv.appendChild(bubbleDiv);
            }});
            
            container.appendChild(pageDiv);
            
            // Update navigation
            document.getElementById('currentPage').textContent = pageIndex + 1;
            document.getElementById('prevBtn').disabled = pageIndex === 0;
            document.getElementById('nextBtn').disabled = pageIndex === pagesData.length - 1;
        }}
        
        function startDrag(e) {{
            isDragging = true;
            dragElement = e.target;
            dragElement.style.zIndex = '1000';
            
            const rect = dragElement.getBoundingClientRect();
            dragOffset.x = e.clientX - rect.left;
            dragOffset.y = e.clientY - rect.top;
            
            document.addEventListener('mousemove', drag);
            document.addEventListener('mouseup', stopDrag);
            
            e.preventDefault();
        }}
        
        function drag(e) {{
            if (!isDragging || !dragElement) return;
            
            const pageRect = document.querySelector('.comic-page').getBoundingClientRect();
            const newX = e.clientX - pageRect.left - dragOffset.x;
            const newY = e.clientY - pageRect.top - dragOffset.y;
            
            // Constrain to page bounds
            const maxX = 1600 - dragElement.offsetWidth;
            const maxY = 1080 - dragElement.offsetHeight;
            
            dragElement.style.left = Math.max(0, Math.min(newX, maxX)) + 'px';
            dragElement.style.top = Math.max(0, Math.min(newY, maxY)) + 'px';
        }}
        
        function stopDrag() {{
            if (dragElement) {{
                dragElement.style.zIndex = '10';
                
                // Update page data
                const pageData = pagesData[currentPageIndex];
                const bubbleIndex = Array.from(document.querySelectorAll('.speech-bubble')).indexOf(dragElement);
                if (bubbleIndex >= 0 && bubbleIndex < pageData.bubbles.length) {{
                    pageData.bubbles[bubbleIndex].x = parseInt(dragElement.style.left);
                    pageData.bubbles[bubbleIndex].y = parseInt(dragElement.style.top);
                }}
            }}
            
            isDragging = false;
            dragElement = null;
            
            document.removeEventListener('mousemove', drag);
            document.removeEventListener('mouseup', stopDrag);
        }}
        
        function editBubble(e) {{
            const bubble = e.target;
            const currentText = bubble.textContent;
            
            bubble.classList.add('editing');
            bubble.contentEditable = true;
            bubble.focus();
            
            // Select all text
            const range = document.createRange();
            range.selectNodeContents(bubble);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            
            bubble.addEventListener('blur', function() {{
                bubble.classList.remove('editing');
                bubble.contentEditable = false;
                
                // Update page data
                const pageData = pagesData[currentPageIndex];
                const bubbleIndex = Array.from(document.querySelectorAll('.speech-bubble')).indexOf(bubble);
                if (bubbleIndex >= 0 && bubbleIndex < pageData.bubbles.length) {{
                    pageData.bubbles[bubbleIndex].text = bubble.textContent;
                }}
            }});
            
            bubble.addEventListener('keydown', function(e) {{
                if (e.key === 'Enter') {{
                    e.preventDefault();
                    bubble.blur();
                }}
            }});
        }}
        
        function previousPage() {{
            if (currentPageIndex > 0) {{
                loadPage(currentPageIndex - 1);
            }}
        }}
        
        function nextPage() {{
            if (currentPageIndex < pagesData.length - 1) {{
                loadPage(currentPageIndex + 1);
            }}
        }}
        
        function setZoom(value) {{
            currentZoom = value / 100;
            document.getElementById('zoomValue').textContent = value + '%';
            loadPage(currentPageIndex);
        }}
        
        function printCurrentPage() {{
            const pageDiv = document.querySelector('.comic-page');
            if (pageDiv) {{
                const printWindow = window.open('', '_blank');
                printWindow.document.write(`
                    <html>
                        <head>
                            <title>Comic Page ${{currentPageIndex + 1}}</title>
                            <style>
                                body {{ margin: 0; padding: 20px; }}
                                .comic-page {{ 
                                    width: 1600px; 
                                    height: 1080px; 
                                    border: 2px solid #000;
                                    position: relative;
                                }}
                                .panel {{ 
                                    position: absolute; 
                                    width: 800px; 
                                    height: 540px; 
                                    border: 1px solid #ddd; 
                                }}
                                .panel img {{ width: 100%; height: 100%; object-fit: cover; }}
                                .speech-bubble {{ 
                                    position: absolute; 
                                    background: white; 
                                    border: 2px solid #333; 
                                    border-radius: 10px; 
                                    padding: 8px 12px; 
                                    font-size: 16px; 
                                    font-weight: bold; 
                                    color: #333; 
                                }}
                            </style>
                        </head>
                        <body>
                            ${{pageDiv.outerHTML}}
                        </body>
                    </html>
                `);
                printWindow.document.close();
                printWindow.print();
            }}
        }}
        
        function saveCurrentPage() {{
            const pageDiv = document.querySelector('.comic-page');
            if (pageDiv) {{
                // Create canvas to render page
                const canvas = document.createElement('canvas');
                canvas.width = 1600;
                canvas.height = 1080;
                const ctx = canvas.getContext('2d');
                
                // Fill white background
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, 1600, 1080);
                
                // Draw border
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 3;
                ctx.strokeRect(1, 1, 1598, 1078);
                
                // Note: This is a simplified version. For full functionality,
                // you would need to render images and text to canvas
                
                // Download as PNG
                const link = document.createElement('a');
                link.download = `comic_page_${{currentPageIndex + 1:03d}}.png`;
                link.href = canvas.toDataURL('image/png');
                link.click();
            }}
        }}
        
        function exportAllPages() {{
            // This would export all 48 pages as PNG files
            alert('Export all pages feature would download 48 PNG files for Unity integration');
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowLeft') previousPage();
            if (e.key === 'ArrowRight') nextPage();
        }});
    </script>
</body>
</html>'''
    
    def _save_unity_data(self, pages_data: List[Dict]):
        """Save data in Unity-friendly format"""
        # Save pages data as JSON
        with open('output/unity_pages/pages_data.json', 'w') as f:
            json.dump(pages_data, f, indent=2)
        
        # Create Unity import instructions
        unity_instructions = """
# Unity Integration Instructions

## Files Generated:
- 48 PNG pages (1600x1080 each) in output/unity_pages/
- Interactive viewer: output/unity_pages/interactive_viewer.html
- Pages data: output/unity_pages/pages_data.json

## Unity Setup:
1. Import all PNG files into Unity as Sprites
2. Set Sprite Import Settings:
   - Texture Type: Sprite (2D and UI)
   - Sprite Mode: Single
   - Pixels Per Unit: 100
   - Filter Mode: Point (no filter)
   - Compression: None (for best quality)

3. Create UI Canvas for comic display
4. Use Image components to display pages
5. Implement page navigation system

## Panel Layout:
- Each page: 1600x1080 pixels
- 2x2 grid: 4 panels per page
- Panel size: 800x540 pixels each
- Total: 48 pages = 192 panels

## Speech Bubbles:
- Draggable in interactive viewer
- Double-click to edit text
- Position data saved in JSON
- Can be recreated in Unity UI system
"""
        
        with open('output/unity_pages/UNITY_INTEGRATION.md', 'w') as f:
            f.write(unity_instructions)

if __name__ == "__main__":
    generator = UnityComicGenerator()
    success = generator.generate_48_pages_comic()
    
    if success:
        print("\n🎉 Unity Comic Generation Complete!")
        print("📁 Check output/unity_pages/ for all files")
        print("🌐 Open output/unity_pages/interactive_viewer.html to view")
        print("📖 Read output/unity_pages/UNITY_INTEGRATION.md for Unity setup")
    else:
        print("\n❌ Unity Comic Generation Failed!")