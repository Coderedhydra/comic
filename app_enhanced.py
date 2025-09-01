"""
Enhanced Comic Generation Application
High-quality comic generation using AI-enhanced processing
"""

import os
import webbrowser
import time
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import srt
import json
import shutil

# Import enhanced modules
from backend.ai_enhanced_core import (
    image_processor, comic_styler, face_detector, layout_optimizer
)
from backend.ai_bubble_placement import ai_bubble_placer
from backend.subtitles.subs_real import get_real_subtitles
from backend.keyframes.keyframes_simple import generate_keyframes_simple
from backend.keyframes.keyframes import black_bar_crop
from backend.class_def import bubble, panel, Page

app = Flask(__name__)

# Import editor routes
try:
    from comic_editor_server import add_editor_routes
    add_editor_routes(app)
    print("✅ Comic editor integrated!")
except Exception as e:
    print(f"⚠️ Could not load comic editor: {e}")

# Ensure directories exist
os.makedirs('video', exist_ok=True)
os.makedirs('frames/final', exist_ok=True)
os.makedirs('output', exist_ok=True)

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
            # 1. Extract real subtitles from video audio
            print("📝 Extracting real subtitles from video...")
            get_real_subtitles(self.video_path)
            
            # 2. Generate keyframes with simplified method (avoids infinite loops)
            print("🎯 Generating keyframes...")
            generate_keyframes_simple(self.video_path)
            
            # 3. Remove black bars
            print("✂️ Removing black bars...")
            black_x, black_y, _, _ = black_bar_crop()
            
            # 4. Enhance image quality with advanced models
            if self.quality_mode == '1':
                print("✨ Enhancing image quality with advanced AI models...")
                self._enhance_all_images_advanced()
            
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
        """Enhance quality of all extracted frames (legacy method)"""
        if not os.path.exists(self.frames_dir):
            print(f"❌ Frames directory not found: {self.frames_dir}")
            return
            
        frame_files = [f for f in os.listdir(self.frames_dir) if f.endswith('.png')]
        print(f"Found {len(frame_files)} frames to enhance")
        
        for i, frame_file in enumerate(frame_files, 1):
            try:
                frame_path = os.path.join(self.frames_dir, frame_file)
                image_processor.enhance_image_quality(frame_path, frame_path)
                print(f"Enhanced: {frame_file} ({i}/{len(frame_files)})")
            except Exception as e:
                print(f"Failed to enhance {frame_file}: {e}")
    
    def _enhance_all_images_advanced(self):
        """Enhance quality using advanced AI models (Real-ESRGAN, GFPGAN, etc.)"""
        if not os.path.exists(self.frames_dir):
            print(f"❌ Frames directory not found: {self.frames_dir}")
            return
        
        try:
            # Get advanced enhancer
            from backend.advanced_image_enhancer import get_advanced_enhancer
            enhancer = get_advanced_enhancer()
            
            frame_files = [f for f in os.listdir(self.frames_dir) if f.endswith('.png')]
            print(f"🚀 Found {len(frame_files)} frames to enhance with advanced AI models")
            
            for i, frame_file in enumerate(frame_files, 1):
                try:
                    frame_path = os.path.join(self.frames_dir, frame_file)
                    print(f"🎯 Enhancing {frame_file} ({i}/{len(frame_files)}) with advanced AI...")
                    
                    # Apply advanced enhancement
                    enhanced_path = enhancer.enhance_image(frame_path, frame_path)
                    
                    if enhanced_path != frame_path:
                        print(f"✅ Advanced enhancement completed: {frame_file}")
                    else:
                        print(f"⚠️ Using fallback enhancement for: {frame_file}")
                        
                except Exception as e:
                    print(f"❌ Advanced enhancement failed for {frame_file}: {e}")
                    # Fallback to basic enhancement
                    try:
                        from backend.ai_enhanced_core import image_processor
                        image_processor.enhance_image_quality(frame_path, frame_path)
                        print(f"🔄 Applied fallback enhancement to: {frame_file}")
                    except Exception as fallback_e:
                        print(f"❌ Fallback enhancement also failed for {frame_file}: {fallback_e}")
                        
        except Exception as e:
            print(f"❌ Advanced enhancement system failed: {e}")
            print("🔄 Falling back to basic enhancement...")
            self._enhance_all_images()
    
    def _apply_comic_styling(self):
        """Apply comic styling to all frames"""
        if not os.path.exists(self.frames_dir):
            print(f"❌ Frames directory not found: {self.frames_dir}")
            return
            
        frame_files = [f for f in os.listdir(self.frames_dir) if f.endswith('.png')]
        print(f"Found {len(frame_files)} frames to style")
        
        for i, frame_file in enumerate(frame_files, 1):
            try:
                frame_path = os.path.join(self.frames_dir, frame_file)
                comic_styler.apply_comic_style(frame_path, frame_path)
                print(f"Styled: {frame_file} ({i}/{len(frame_files)})")
            except Exception as e:
                print(f"Failed to style {frame_file}: {e}")
    
    def _generate_optimized_layout(self):
        """Generate AI-optimized layout"""
        try:
            layout_data = layout_optimizer.optimize_layout(self.frames_dir)
            return layout_data
        except Exception as e:
            print(f"Layout generation failed: {e}")
            # Fallback to simple 2x2 layout
            return ['6666', '6666', '6666', '6666']
    
    def _create_ai_bubbles(self, black_x, black_y):
        """Create AI-powered speech bubbles"""
        bubbles = []
        
        try:
            # Read subtitles
            with open('test1.srt', 'r', encoding='utf-8') as f:
                subs = list(srt.parse(f.read()))
            
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            
            for i, frame_file in enumerate(frame_files):
                if i < len(subs):
                    sub = subs[i]
                    frame_path = os.path.join(self.frames_dir, frame_file)
                    
                    try:
                        # Get lip coordinates (simplified)
                        lip_x, lip_y = -1, -1
                        
                        # Try to detect faces and get lip position
                        try:
                            faces = face_detector.detect_faces(frame_path)
                            if faces:
                                lip_x, lip_y = face_detector.get_lip_position(frame_path, faces[0])
                        except Exception as e:
                            print(f"Face detection failed for {frame_file}: {e}")
                        
                        print(f"lipx = {lip_x} and lipy = {lip_y}")
                        
                        # Get bubble position using AI
                        bubble_x, bubble_y = ai_bubble_placer.place_bubble_ai(
                            frame_path, (lip_x, lip_y)
                        )
                        
                        # Create bubble
                        bubble_obj = bubble(
                            bubble_offset_x=bubble_x,
                            bubble_offset_y=bubble_y,
                            lip_x=lip_x,
                            lip_y=lip_y,
                            dialog=sub.content,
                            emotion='normal'
                        )
                        
                        bubbles.append(bubble_obj)
                        
                    except Exception as e:
                        print(f"Bubble creation failed for {frame_file}: {e}")
                        # Create fallback bubble
                        bubble_obj = bubble(
                            bubble_offset_x=50,
                            bubble_offset_y=50,
                            lip_x=-1,
                            lip_y=-1,
                            dialog=sub.content,
                            emotion='normal'
                        )
                        bubbles.append(bubble_obj)
                        
        except Exception as e:
            print(f"Bubble creation failed: {e}")
        
        return bubbles
    
    def _generate_pages(self, layout_data, bubbles):
        """Generate final pages"""
        pages = []
        
        try:
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            
            # Create 4 pages with different frame combinations
            for page_num in range(4):
                # Create panels for this page
                panels = []
                page_bubbles = []
                
                for j in range(4):
                    # Use unique frames for each page (16 total frames, 4 per page)
                    frame_index = page_num * 4 + j
                    if frame_index < len(frame_files):
                        frame_file = frame_files[frame_index]
                    else:
                        # Fallback to first frame if we don't have enough
                        frame_file = frame_files[0]
                    
                    panel_obj = panel(
                        image=frame_file,
                        row_span=2,
                        col_span=2
                    )
                    panels.append(panel_obj)
                    
                    # Add corresponding bubble with correct dialogue
                    bubble_index = page_num * 4 + j
                    if bubble_index < len(bubbles):
                        # Use the correct bubble for this frame
                        original_bubble = bubbles[bubble_index]
                        bubble_obj = bubble(
                            bubble_offset_x=original_bubble.bubble_offset_x,
                            bubble_offset_y=original_bubble.bubble_offset_y,
                            lip_x=-1,  # Use default values
                            lip_y=-1,  # Use default values
                            dialog=original_bubble.dialog,  # Use original dialogue
                            emotion=original_bubble.emotion
                        )
                        page_bubbles.append(bubble_obj)
                    else:
                        # Create fallback bubble with varied dialogue
                        fallback_dialogues = [
                            "Hello there!", "How are you doing?", "I'm doing great!", "That's wonderful!",
                            "What's new?", "Not much, just working.", "Sounds busy!", "It sure is!",
                            "Any plans for today?", "Just relaxing.", "That sounds nice!", "Indeed it is.",
                            "Have a great day!", "You too!", "See you later!", "Take care!"
                        ]
                        fallback_dialog = fallback_dialogues[bubble_index % len(fallback_dialogues)]
                        fallback_bubble = bubble(
                            bubble_offset_x=50,
                            bubble_offset_y=200,
                            lip_x=-1,
                            lip_y=-1,
                            dialog=fallback_dialog,
                            emotion='normal'
                        )
                        page_bubbles.append(fallback_bubble)
                
                # Create page
                page = Page(panels=panels, bubbles=page_bubbles)
                pages.append(page)
                
        except Exception as e:
            print(f"Page generation failed: {e}")
        
        return pages
    
    def _save_results(self, pages):
        """Save results to output directory"""
        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save pages data
            pages_data = []
            for page in pages:
                page_data = {
                    'panels': page.panels,  # Already dictionaries
                    'bubbles': page.bubbles  # Already dictionaries
                }
                pages_data.append(page_data)
            
            with open(os.path.join(self.output_dir, 'pages.json'), 'w') as f:
                json.dump(pages_data, f, indent=2)
            
            # Copy template files
            self._copy_template_files()
            
            print("✅ Results saved successfully!")
            print(f"📁 Comic saved to: {os.path.abspath('output/page.html')}")
            
        except Exception as e:
            print(f"Save results failed: {e}")
    
    def _copy_template_files(self):
        """Copy template files to output directory"""
        try:
            # Copy HTML template
            template_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Comic</title>
    <style>
        body { margin: 0; padding: 20px; background: #f0f0f0; font-family: Arial, sans-serif; }
        .comic-container { max-width: 1200px; margin: 0 auto; }
        .comic-page { background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .comic-grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 10px; height: 600px; }
        .page-title { text-align: center; color: #333; margin-bottom: 15px; font-size: 18px; font-weight: bold; }
        .panel { position: relative; border: 2px solid #333; overflow: hidden; }
        .panel img { width: 100%; height: 100%; object-fit: cover; }
        .speech-bubble { 
            position: absolute; 
            background: white; 
            border: 3px solid #333; 
            border-radius: 15px; 
            padding: 12px; 
            max-width: 200px; 
            font-size: 14px; 
            font-weight: bold;
            box-shadow: 3px 3px 8px rgba(0,0,0,0.4);
            z-index: 10;
            text-align: center;
            color: #333;
        }
        .speech-bubble::after { 
            content: ''; 
            position: absolute; 
            bottom: -10px; 
            left: 20px; 
            width: 0; 
            height: 0; 
            border-left: 10px solid transparent; 
            border-right: 10px solid transparent; 
            border-top: 10px solid #333; 
        }
        .comic-title { text-align: center; color: #333; margin-bottom: 20px; }
        .loading { text-align: center; color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="comic-container">
        <h1 class="comic-title">🎬 Generated Comic</h1>
        <div id="comic-pages">
            <div class="loading">Loading comic...</div>
        </div>
    </div>
    <script>
        // Load comic data
        fetch('/output/pages.json')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load pages.json');
                }
                return response.json();
            })
            .then(data => {
                const pagesContainer = document.getElementById('comic-pages');
                pagesContainer.innerHTML = ''; // Clear loading message
                
                if (data && data.length > 0) {
                    // Create multiple pages
                    data.forEach((pageData, pageIndex) => {
                        if (pageData.panels && pageData.panels.length > 0) {
                            // Create page container
                            const pageDiv = document.createElement('div');
                            pageDiv.className = 'comic-page';
                            
                            // Add page title
                            const pageTitle = document.createElement('h2');
                            pageTitle.className = 'page-title';
                            pageTitle.textContent = `Page ${pageIndex + 1}`;
                            pageDiv.appendChild(pageTitle);
                            
                            // Create grid for this page
                            const grid = document.createElement('div');
                            grid.className = 'comic-grid';
                            
                            // Add panels to this page
                            pageData.panels.forEach((panel, index) => {
                                const panelDiv = document.createElement('div');
                                panelDiv.className = 'panel';
                                
                                const img = document.createElement('img');
                                img.src = '/frames/final/' + panel.image;
                                img.alt = `Page ${pageIndex + 1} - Panel ${index + 1}`;
                                img.onerror = function() {
                                    this.style.display = 'none';
                                    panelDiv.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #666;">Image not found</div>';
                                };
                                panelDiv.appendChild(img);
                                
                                // Add speech bubbles
                                if (pageData.bubbles && pageData.bubbles[index]) {
                                    const bubble = pageData.bubbles[index];
                                    const bubbleDiv = document.createElement('div');
                                    bubbleDiv.className = 'speech-bubble';
                                    
                                    // Use bubble_offset_x and bubble_offset_y from the data
                                    // Fix positioning - ensure bubbles are visible within panel
                                    let x = bubble.bubble_offset_x || 50;
                                    let y = bubble.bubble_offset_y || 50;
                                    
                                    // Clamp positions to ensure bubbles are visible
                                    x = Math.max(10, Math.min(x, 300));
                                    y = Math.max(10, Math.min(y, 200));
                                    
                                    bubbleDiv.style.left = x + 'px';
                                    bubbleDiv.style.top = y + 'px';
                                    bubbleDiv.style.maxWidth = '180px';
                                    bubbleDiv.style.minHeight = '50px';
                                    bubbleDiv.style.fontSize = '12px';
                                    bubbleDiv.style.lineHeight = '1.2';
                                    bubbleDiv.style.wordWrap = 'break-word';
                                    
                                    // Use dialog from the data
                                    bubbleDiv.textContent = bubble.dialog || '((action-scene))';
                                    panelDiv.appendChild(bubbleDiv);
                                }
                                
                                grid.appendChild(panelDiv);
                            });
                            
                            pageDiv.appendChild(grid);
                            pagesContainer.appendChild(pageDiv);
                        }
                    });
                } else {
                    pagesContainer.innerHTML = '<div class="loading">No comic data found</div>';
                }
            })
            .catch(error => {
                console.error('Error loading comic:', error);
                document.getElementById('comic-grid').innerHTML = '<div class="loading">Error loading comic data: ' + error.message + '</div>';
            });
    </script>
</body>
</html>'''
            
            with open(os.path.join(self.output_dir, 'page.html'), 'w') as f:
                f.write(template_html)
            
            print("📄 Template files copied successfully!")
            
        except Exception as e:
            print(f"Template copy failed: {e}")

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
            
            if 'file' not in request.files:
                return "❌ No file uploaded"
            
            f = request.files['file']
            if f.filename == '':
                return "❌ No file selected"
            
            # Clean up previous files
            if os.path.exists('video/uploaded.mp4'):
                os.remove('video/uploaded.mp4')
            
            # Save uploaded file
            f.save("video/uploaded.mp4")
            print(f"✅ File saved: {f.filename}")
            
            # Generate comic
            success = comic_generator.generate_comic()
            
            if success:
                # Open result in browser through Flask
                comic_url = f"http://localhost:5000/comic"
                print(f"🌐 Opening comic in browser: {comic_url}")
                try:
                    webbrowser.open(comic_url)
                    print("✅ Browser opened successfully!")
                except Exception as e:
                    print(f"⚠️ Could not open browser: {e}")
                    print(f"📁 Please open manually: {comic_url}")
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
            link = request.form.get('link', '')
            
            if not link:
                return "❌ No link provided"
            
            # Clean up previous files
            if os.path.exists('video/uploaded.mp4'):
                os.remove('video/uploaded.mp4')
            
            # Download video using yt-dlp
            try:
                import yt_dlp
                ydl_opts = {
                    'outtmpl': 'video/uploaded.mp4',
                    'format': 'best[height<=720]'
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([link])
                print(f"✅ Video downloaded from: {link}")
            except Exception as e:
                print(f"❌ Video download failed: {e}")
                return f"❌ Failed to download video: {str(e)}"
            
            # Generate comic
            success = comic_generator.generate_comic()
            
            if success:
                # Open result in browser through Flask
                comic_url = f"http://localhost:5000/comic"
                print(f"🌐 Opening comic in browser: {comic_url}")
                try:
                    webbrowser.open(comic_url)
                    print("✅ Browser opened successfully!")
                except Exception as e:
                    print(f"⚠️ Could not open browser: {e}")
                    print(f"📁 Please open manually: {comic_url}")
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
        'frames_exist': os.path.exists(comic_generator.frames_dir),
        'output_exists': os.path.exists('output/page.html')
    })

@app.route('/output/<path:filename>')
def output_file(filename):
    """Serve output files"""
    return send_from_directory('output', filename)

@app.route('/frames/final/<path:filename>')
def frame_file(filename):
    """Serve frame files"""
    return send_from_directory('frames/final', filename)

@app.route('/comic')
def view_comic():
    """Serve the generated comic page"""
    return send_from_directory('output', 'page.html')

if __name__ == '__main__':
    print("🚀 Starting Enhanced Comic Generator...")
    print("✨ Features:")
    print("   - AI-enhanced image processing")
    print("   - Advanced face detection")
    print("   - Smart bubble placement")
    print("   - High-quality comic styling")
    print("   - Optimized 2x2 layout")
    print("")
    print("🌐 Web interface available at: http://localhost:5000")
    print("📁 Upload videos or paste YouTube links to generate comics!")
    print("")
    app.run(debug=True, host='0.0.0.0', port=5000)