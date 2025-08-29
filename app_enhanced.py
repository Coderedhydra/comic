"""
Enhanced Comic Generation Application
High-quality comic generation using AI-enhanced processing
"""

import os
import webbrowser
import time
import threading
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import srt
import json

# Import enhanced modules
from backend.ai_enhanced_core import (
    image_processor, comic_styler, face_detector, layout_optimizer
)
from backend.ai_bubble_placement import ai_bubble_placer
from backend.subtitles.subs import get_subtitles
from backend.keyframes.keyframes import generate_keyframes, black_bar_crop
from backend.class_def import bubble, panel, Page
from backend.utils import cleanup, download_video, copy_template

app = Flask(__name__)

class EnhancedComicGenerator:
    """High-quality comic generation with AI enhancement"""
    
    def __init__(self):
        self.video_path = 'video/uploaded.mp4'
        self.frames_dir = 'frames/final'
        self.output_dir = 'output'
        self.quality_mode = os.getenv('HIGH_QUALITY', '1')
        self.ai_mode = os.getenv('AI_ENHANCED', '1')
        
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                print("🚀 GPU detected! Using CUDA acceleration")
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            else:
                print("💻 Using CPU processing")
        except:
            print("💻 Using CPU processing")
        
    def generate_comic(self):
        """Main comic generation pipeline"""
        start_time = time.time()
        
        print("🎬 Starting Enhanced Comic Generation...")
        
        try:
            # 1. Extract subtitles
            print("📝 Extracting subtitles...")
            get_subtitles(self.video_path)
            
            # 2. Generate keyframes with enhanced quality
            print("🎯 Generating high-quality keyframes...")
            generate_keyframes(self.video_path)
            
            # 3. Remove black bars
            print("✂️ Removing black bars...")
            black_x, black_y, _, _ = black_bar_crop()
            
            # 4. Enhance image quality
            if self.quality_mode == '1':
                print("✨ Enhancing image quality...")
                self._enhance_all_images()
            
            # 5. Apply comic styling
            print("🎨 Applying AI-enhanced comic styling...")
            self._apply_comic_styling()
            
            # 6. Generate optimized layout
            print("📐 Generating AI-optimized layout...")
            layout_data = self._generate_optimized_layout()
            
            # 7. Create AI-powered speech bubbles
            print("💬 Creating AI-powered speech bubbles...")
            bubbles = self._create_ai_bubbles(black_x, black_y)
            
            # 8. Generate final pages
            print("📄 Generating final pages...")
            pages = self._generate_pages(layout_data, bubbles)
            
            # 9. Save results
            print("💾 Saving results...")
            self._save_results(pages)
            
            execution_time = (time.time() - start_time) / 60
            print(f"✅ Comic generation completed in {execution_time:.2f} minutes")
            
            return True
            
        except Exception as e:
            print(f"❌ Comic generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _enhance_all_images(self):
        """Enhance quality of all extracted frames"""
        if not os.path.exists(self.frames_dir):
            print("No frames directory found, skipping enhancement")
            return
        
        frame_files = [f for f in os.listdir(self.frames_dir) if f.endswith('.png')]
        print(f"Found {len(frame_files)} frames to enhance")
        
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(self.frames_dir, frame_file)
            try:
                # Apply AI-enhanced image processing
                image_processor.enhance_image_quality(frame_path)
                print(f"Enhanced: {frame_file} ({i+1}/{len(frame_files)})")
            except Exception as e:
                print(f"Failed to enhance {frame_file}: {e}")
                # Continue with other frames
                continue
    
    def _apply_comic_styling(self):
        """Apply AI-enhanced comic styling to all frames"""
        if not os.path.exists(self.frames_dir):
            print("No frames directory found, skipping styling")
            return
        
        frame_files = [f for f in os.listdir(self.frames_dir) if f.endswith('.png')]
        print(f"Found {len(frame_files)} frames to style")
        
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(self.frames_dir, frame_file)
            try:
                # Apply modern comic style
                comic_styler.apply_comic_style(frame_path, style_type="modern")
                print(f"Styled: {frame_file} ({i+1}/{len(frame_files)})")
            except Exception as e:
                print(f"Failed to style {frame_file}: {e}")
                # Continue with other frames
                continue
    
    def _generate_optimized_layout(self):
        """Generate AI-optimized layout"""
        if not os.path.exists(self.frames_dir):
            return {'panels': [], 'templates': []}
        
        frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        frame_paths = [os.path.join(self.frames_dir, f) for f in frame_files]
        
        # Use AI layout optimizer
        if self.ai_mode == '1':
            layout = layout_optimizer.optimize_layout(frame_paths, target_layout="2x2")
        else:
            # Fallback to simple 2x2 layout
            layout = self._create_simple_layout(frame_paths)
        
        return {
            'panels': layout,
            'templates': ['6666'] * (len(frame_paths) // 4 + 1)
        }
    
    def _create_simple_layout(self, frame_paths):
        """Create simple 2x2 layout as fallback"""
        layout = []
        for i, frame_path in enumerate(frame_paths[:4]):  # Limit to 4 images
            layout.append({
                'index': i,
                'type': '6',
                'span': (2, 2),
                'priority': 'medium',
                'content_analysis': {'complexity': 'medium', 'faces': 0, 'action': 'low'}
            })
        return layout
    
    def _create_ai_bubbles(self, black_x, black_y):
        """Create AI-powered speech bubbles"""
        bubbles = []
        
        # Read subtitles
        with open("test1.srt") as f:
            data = f.read()
        subs = srt.parse(data)
        
        # Get lip positions using AI face detection
        lip_positions = self._get_ai_lip_positions()
        
        for sub in subs:
            if sub.content == "((action-scene))":
                continue
            
            # Get lip coordinates
            lip_x, lip_y = lip_positions.get(sub.index, (-1, -1))
            
            # Get panel coordinates (simplified for 2x2 layout)
            panel_coords = self._get_panel_coords(sub.index)
            
            # Use AI bubble placement
            if self.ai_mode == '1':
                bubble_x, bubble_y = ai_bubble_placer.place_bubble_ai(
                    f"frames/final/frame{sub.index:03}.png",
                    panel_coords,
                    (lip_x, lip_y),
                    sub.content
                )
            else:
                # Fallback positioning
                bubble_x, bubble_y = self._get_fallback_position(panel_coords, lip_x, lip_y)
            
            # Determine bubble shape based on content
            bubble_shape = self._determine_bubble_shape(sub.content)
            
            # Create bubble object
            bubble_obj = bubble(bubble_x, bubble_y, lip_x, lip_y, sub.content, bubble_shape)
            bubbles.append(bubble_obj)
        
        return bubbles
    
    def _get_ai_lip_positions(self):
        """Get lip positions using AI face detection"""
        lip_positions = {}
        
        if not os.path.exists(self.frames_dir):
            return lip_positions
        
        frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        
        for frame_file in frame_files:
            frame_path = os.path.join(self.frames_dir, frame_file)
            frame_index = int(frame_file.replace('frame', '').replace('.png', ''))
            
            try:
                # Use AI face detection
                faces = face_detector.detect_faces_advanced(frame_path)
                
                if faces:
                    # Use the first detected face
                    lip_pos = faces[0]['lip_position']
                    lip_positions[frame_index] = lip_pos
                else:
                    lip_positions[frame_index] = (-1, -1)
                    
            except Exception as e:
                print(f"Face detection failed for {frame_file}: {e}")
                lip_positions[frame_index] = (-1, -1)
        
        return lip_positions
    
    def _get_panel_coords(self, frame_index):
        """Get panel coordinates for 2x2 layout"""
        # Simplified 2x2 layout coordinates
        panel_width = 1035 // 2
        panel_height = 1100 // 2
        
        # Calculate panel position based on frame index
        row = (frame_index - 1) // 2
        col = (frame_index - 1) % 2
        
        left = col * panel_width
        right = (col + 1) * panel_width
        top = row * panel_height
        bottom = (row + 1) * panel_height
        
        return (left, right, top, bottom)
    
    def _get_fallback_position(self, panel_coords, lip_x, lip_y):
        """Get fallback bubble position"""
        left, right, top, bottom = panel_coords
        
        # Simple upper-right positioning
        bubble_x = right - 250  # 50px margin from right
        bubble_y = top + 50     # 50px margin from top
        
        # Avoid lip if detected
        if lip_x != -1 and lip_y != -1:
            distance = ((bubble_x - lip_x)**2 + (bubble_y - lip_y)**2)**0.5
            if distance < 100:
                bubble_x = left + 100
                bubble_y = top + 100
        
        return bubble_x, bubble_y
    
    def _determine_bubble_shape(self, dialogue):
        """Determine bubble shape based on dialogue content"""
        dialogue_lower = dialogue.lower()
        
        # Check for exclamations
        if any(word in dialogue_lower for word in ['!', 'wow', 'oh', 'ah', 'hey']):
            return 'exclamation'
        
        # Check for questions
        if any(word in dialogue_lower for word in ['?', 'what', 'how', 'why', 'when']):
            return 'question'
        
        # Check for thoughts
        if any(word in dialogue_lower for word in ['think', 'thought', 'maybe', 'perhaps']):
            return 'thought'
        
        # Default to normal speech
        return 'normal'
    
    def _generate_pages(self, layout_data, bubbles):
        """Generate final pages"""
        pages = []
        
        # Group bubbles by page (4 per page for 2x2 layout)
        bubbles_per_page = 4
        for i in range(0, len(bubbles), bubbles_per_page):
            page_bubbles = bubbles[i:i + bubbles_per_page]
            
            # Create panels for this page
            panels = []
            for j in range(min(4, len(page_bubbles))):
                panel_obj = panel(
                    left=0, right=1035//2, top=0, bottom=1100//2,
                    span=(2, 2), type='6'
                )
                panels.append(panel_obj)
            
            # Create page
            page = Page(panels, page_bubbles)
            pages.append(page)
        
        return pages
    
    def _save_results(self, pages):
        """Save final results"""
        print("💾 Saving results...")
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Save pages as JSON
        pages_data = []
        for page in pages:
            page_data = {
                'panels': [{'left': p.left, 'right': p.right, 'top': p.top, 'bottom': p.bottom} for p in page.panels],
                'bubbles': [{'x': b.x, 'y': b.y, 'content': b.content, 'shape': b.shape} for b in page.bubbles]
            }
            pages_data.append(page_data)
        
        with open('output/pages.json', 'w') as f:
            json.dump(pages_data, f, indent=2)
        
        print("📄 Copying template files...")
        # Copy template files
        copy_template()
        
        print("✅ Results saved successfully!")
        print(f"📁 Comic saved to: {os.path.abspath('output/page.html')}")

# Global comic generator instance
comic_generator = EnhancedComicGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            print("📁 Processing file upload...")
            f = request.files['file']
            
            # Clean up previous files
            cleanup()
            
            # Save uploaded file
            f.save("video/uploaded.mp4")
            
            # Generate comic
            success = comic_generator.generate_comic()
            
            if success:
                # Open result in browser
                output_path = os.path.join(os.getcwd(), 'output', 'page.html')
                print(f"🌐 Opening comic in browser: {output_path}")
                try:
                    webbrowser.open(f'file://{output_path}')
                    print("✅ Browser opened successfully!")
                except Exception as e:
                    print(f"⚠️ Could not open browser: {e}")
                    print(f"📁 Please open manually: {output_path}")
                return "🎉 Enhanced Comic Created Successfully!"
            else:
                return "❌ Comic generation failed"
                
        except Exception as e:
            print(f"Error during comic generation: {e}")
            return f"❌ Error: {str(e)}"

@app.route('/handle_link', methods=['GET', 'POST'])
def handle_link():
    if request.method == 'POST':
        try:
            print("🔗 Processing video link...")
            link = request.form['link']
            
            # Clean up previous files
            cleanup()
            
            # Download video
            download_video(link)
            
            # Generate comic
            success = comic_generator.generate_comic()
            
            if success:
                # Open result in browser
                output_path = os.path.join(os.getcwd(), 'output', 'page.html')
                print(f"🌐 Opening comic in browser: {output_path}")
                try:
                    webbrowser.open(f'file://{output_path}')
                    print("✅ Browser opened successfully!")
                except Exception as e:
                    print(f"⚠️ Could not open browser: {e}")
                    print(f"📁 Please open manually: {output_path}")
                return "🎉 Enhanced Comic Created Successfully!"
            else:
                return "❌ Comic generation failed"
                
        except Exception as e:
            print(f"Error during comic generation: {e}")
            return f"❌ Error: {str(e)}"

@app.route('/status')
def status():
    """Return system status"""
    return jsonify({
        'ai_mode': comic_generator.ai_mode,
        'quality_mode': comic_generator.quality_mode,
        'video_exists': os.path.exists(comic_generator.video_path),
        'frames_exist': os.path.exists(comic_generator.frames_dir)
    })

if __name__ == '__main__':
    print("🚀 Starting Enhanced Comic Generator...")
    print("✨ Features:")
    print("   - AI-enhanced image processing")
    print("   - Advanced face detection")
    print("   - Smart bubble placement")
    print("   - High-quality comic styling")
    print("   - Optimized 2x2 layout")
    print("")
    app.run(debug=True, host='0.0.0.0', port=5000)