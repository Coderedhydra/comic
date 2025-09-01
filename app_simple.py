"""
Simple Comic Generator App
- NO comic styling (preserves colors)
- ONLY 12 meaningful story panels
- Clean grid layout
"""

import os
import time
from flask import Flask, request, render_template, send_from_directory
from backend.subtitles.subs_real import get_real_subtitles
from backend.simple_comic_generator import SimpleComicGenerator
from backend.advanced_image_enhancer import AdvancedImageEnhancer

app = Flask(__name__)

# Ensure directories exist
os.makedirs('video', exist_ok=True)
os.makedirs('frames/final', exist_ok=True)
os.makedirs('output', exist_ok=True)

class CleanComicGenerator:
    def __init__(self):
        self.video_path = 'video/uploaded.mp4'
        self.simple_generator = SimpleComicGenerator()
        self.enhancer = AdvancedImageEnhancer()
        
    def generate(self):
        """Generate clean comic with meaningful panels only"""
        start_time = time.time()
        
        try:
            print("🎬 Starting Clean Comic Generation...")
            print("📋 Settings:")
            print("  - Target: 12 meaningful panels")
            print("  - No comic styling (preserve colors)")
            print("  - Grid layout: 3x4")
            
            # 1. Extract subtitles
            print("\n📝 Extracting subtitles...")
            get_real_subtitles(self.video_path)
            
            # 2. Generate comic with meaningful panels
            print("\n📖 Selecting meaningful story moments...")
            success = self.simple_generator.generate_meaningful_comic(self.video_path)
            
            if success:
                # 3. Enhance images (optional, preserves colors)
                print("\n✨ Enhancing image quality...")
                self._enhance_frames()
                
                print(f"\n✅ Comic generated in {time.time() - start_time:.1f} seconds!")
                print("📁 View at: output/comic_simple.html")
                return True
            else:
                print("❌ Comic generation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def _enhance_frames(self):
        """Enhance frames with color preservation"""
        frames_dir = 'frames/final'
        frames = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        
        # Configure enhancer for color preservation
        self.enhancer.use_ai_models = False  # Disable AI models that might change colors
        
        for i, frame in enumerate(frames):
            try:
                frame_path = os.path.join(frames_dir, frame)
                print(f"  Enhancing {frame} ({i+1}/{len(frames)})...")
                
                # Basic enhancement only (sharpness, brightness)
                import cv2
                img = cv2.imread(frame_path)
                
                # Slight sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 1
                sharpened = cv2.filter2D(img, -1, kernel)
                
                # Blend with original (preserve colors)
                enhanced = cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)
                
                # Save with high quality
                cv2.imwrite(frame_path, enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
            except Exception as e:
                print(f"  ⚠️ Enhancement failed for {frame}: {e}")

# Global generator instance
comic_generator = CleanComicGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return "❌ No file uploaded"
            
            f = request.files['file']
            if f.filename == '':
                return "❌ No file selected"
            
            # Save video
            f.save("video/uploaded.mp4")
            print(f"✅ Video saved: {f.filename}")
            
            # Generate comic
            success = comic_generator.generate()
            
            if success:
                return '''
                <html>
                <body style="font-family: Arial; padding: 20px;">
                    <h2>✅ Comic Generated Successfully!</h2>
                    <p>Created 12 meaningful story panels with preserved colors.</p>
                    <a href="/comic" style="display: inline-block; padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">View Comic</a>
                </body>
                </html>
                '''
            else:
                return "❌ Comic generation failed"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"

@app.route('/comic')
def view_comic():
    """Serve the generated comic"""
    return send_from_directory('output', 'comic_simple.html')

@app.route('/frames/final/<path:filename>')
def serve_frame(filename):
    """Serve frame images"""
    return send_from_directory('frames/final', filename)

if __name__ == '__main__':
    import numpy as np  # Import for enhancement
    
    print("🚀 Starting Simple Comic Generator...")
    print("✨ Features:")
    print("   - 12 meaningful story panels")
    print("   - Original colors preserved")
    print("   - Clean grid layout")
    print("   - No unnecessary processing")
    print("\n🌐 Open browser to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)