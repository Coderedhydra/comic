import os
import webbrowser
import time
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import srt
import json
import shutil
from typing import List
import traceback

# Import enhanced modules
try:
    from backend.ai_enhanced_core import (
        image_processor, comic_styler, face_detector, layout_optimizer
    )
    from backend.ai_bubble_placement import ai_bubble_placer
    from backend.subtitles.subs_real import get_real_subtitles
    from backend.keyframes.keyframes_simple import generate_keyframes_simple
    from backend.keyframes.keyframes import black_bar_crop
    from backend.class_def import bubble, panel, Page
    from backend.simple_color_enhancer import SimpleColorEnhancer
    from backend.quality_color_enhancer import QualityColorEnhancer
    print("✅ Core modules loaded.")
except Exception as e:
    print(f"⚠️ Could not load a core module: {e}")

# Import smart comic generation
try:
    from backend.emotion_aware_comic import EmotionAwareComicGenerator
    from backend.story_analyzer import SmartComicGenerator
    SMART_COMIC_AVAILABLE = True
    print("✅ Smart comic generation available!")
except Exception as e:
    SMART_COMIC_AVAILABLE = False
    print(f"⚠️ Smart comic generation not available: {e}")

# Import panel extractor
try:
    from backend.panel_extractor import PanelExtractor
    PANEL_EXTRACTOR_AVAILABLE = True
    print("✅ Panel extractor available!")
except Exception as e:
    PANEL_EXTRACTOR_AVAILABLE = False
    print(f"⚠️ Panel extractor not available: {e}")

# Import smart story extractor
try:
    from backend.smart_story_extractor import SmartStoryExtractor
    STORY_EXTRACTOR_AVAILABLE = True
    print("✅ Smart story extractor available!")
except Exception as e:
    STORY_EXTRACTOR_AVAILABLE = False
    print(f"⚠️ Smart story extractor not available: {e}")

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
        self.apply_comic_style = False

    def cleanup_generated(self):
        """Deletes all old files to ensure a fresh start."""
        print("🧹 Performing full cleanup of previous run...")
        if os.path.isdir(self.frames_dir): shutil.rmtree(self.frames_dir)
        if os.path.isdir(self.output_dir): shutil.rmtree(self.output_dir)
        if os.path.isdir('temp'): shutil.rmtree('temp')
        if os.path.exists('test1.srt'): os.remove('test1.srt')
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        print("✅ Cleanup complete.")

    def detect_eye_state(self, frame_path):
        """
        Detect if eyes are closed or semi-closed in a frame
        Returns: 'open', 'semi-closed', or 'closed'
        """
        try:
            img = cv2.imread(frame_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) == 0:
                    return 'closed'
                elif len(eyes) == 1:
                    return 'semi-closed'
                for (ex, ey, ew, eh) in eyes:
                    eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                    vert_var = np.var(eye_region, axis=0).mean()
                    if vert_var < 500:
                        return 'semi-closed'
            return 'open'
        except:
            return 'open'

    def regenerate_frame(self, frame_filename):
        """
        Regenerate frame by moving +0.1s forward in the original video.
        Updates metadata so repeated clicks keep advancing.
        """
        try:
            metadata_path = 'frames/frame_metadata.json'
            if not os.path.exists(metadata_path):
                return {"success": False, "message": "Frame metadata missing."}

            with open(metadata_path, 'r') as f:
                frame_to_time = json.load(f)

            if frame_filename not in frame_to_time:
                return {"success": False, "message": "Panel not linked to original video."}

            # Fix: Handle the new metadata structure
            if isinstance(frame_to_time[frame_filename], dict):
                current_time = frame_to_time[frame_filename]['time']
            else:
                current_time = frame_to_time[frame_filename]
                
            target_time = current_time + 0.1

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return {"success": False, "message": "Cannot open video."}

            cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return {"success": False, "message": "No next frame available at +0.1s."}

            new_path = os.path.join(self.frames_dir, frame_filename)
            cv2.imwrite(new_path, frame)
            
            # Update metadata with new time
            if isinstance(frame_to_time[frame_filename], dict):
                frame_to_time[frame_filename]['time'] = target_time
            else:
                frame_to_time[frame_filename] = target_time
                
            with open(metadata_path, 'w') as f:
                json.dump(frame_to_time, f, indent=2)
            
            print(f"✅ Regenerated {frame_filename} to time {target_time:.2f}s without enhancement.")

            return {
                "success": True,
                "message": f"Advanced to {target_time:.2f}s (+0.1s)",
                "new_filename": frame_filename
            }

        except Exception as e:
            traceback.print_exc()
            return {"success": False, "message": str(e)}

    def generate_keyframes_from_moments(self, video_path, key_moments, max_frames=48):
        """
        Generate frames specifically at the key moments timestamps
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("❌ Cannot open video for keyframe extraction")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # Sort key moments by start time to maintain chronological order
            key_moments.sort(key=lambda x: x['start'])
            
            # Limit to max_frames while preserving story flow
            if len(key_moments) > max_frames:
                # Use a more intelligent sampling to preserve story flow
                # Take first few moments, then sample evenly, then last few moments
                first_count = min(5, max_frames // 4)
                last_count = min(5, max_frames // 4)
                middle_count = max_frames - first_count - last_count
                
                if middle_count > 0:
                    first_moments = key_moments[:first_count]
                    last_moments = key_moments[-last_count:]
                    middle_moments = key_moments[first_count:-last_count]
                    
                    # Sample evenly from middle moments
                    if len(middle_moments) > middle_count:
                        step = len(middle_moments) / middle_count
                        middle_sampled = [middle_moments[int(i * step)] for i in range(middle_count)]
                    else:
                        middle_sampled = middle_moments
                    
                    key_moments = first_moments + middle_sampled + last_moments
                else:
                    # Just take evenly spaced moments
                    step = len(key_moments) / max_frames
                    key_moments = [key_moments[int(i * step)] for i in range(max_frames)]
            
            frame_metadata = {}
            frame_count = 0
            
            for moment in key_moments:
                # Use the middle of the subtitle segment as the frame time
                frame_time = (moment['start'] + moment['end']) / 2
                
                # Skip if beyond video duration
                if frame_time > duration:
                    continue
                
                # Calculate frame number
                frame_number = int(frame_time * fps)
                
                # Set position and extract frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    frame_filename = f"frame_{frame_count:04d}.png"
                    frame_path = os.path.join(self.frames_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    
                    # Store metadata for this frame
                    frame_metadata[frame_filename] = {
                        'time': frame_time,
                        'dialogue': moment['text'],
                        'start': moment['start'],
                        'end': moment['end']
                    }
                    frame_count += 1
                    print(f"📸 Extracted frame at {frame_time:.2f}s: {moment['text'][:30]}...")
            
            cap.release()
            
            # Save frame metadata with dialogue
            with open(os.path.join('frames', 'frame_metadata.json'), 'w') as f:
                json.dump(frame_metadata, f, indent=2)
            
            print(f"✅ Extracted {frame_count} keyframes from video")
            return True
            
        except Exception as e:
            print(f"❌ Error extracting keyframes: {e}")
            traceback.print_exc()
            return False

    def generate_comic(self, smart_mode=False, emotion_match=False):
        """Main comic generation pipeline"""
        start_time = time.time()
        self.cleanup_generated()
        print("🎬 Starting Enhanced Comic Generation...")
        try:
            print("📝 Generating subtitles...")
            get_real_subtitles(self.video_path)
            all_subs = []
            if os.path.exists('test1.srt'):
                with open('test1.srt', 'r', encoding='utf-8') as f:
                    all_subs = list(srt.parse(f.read()))
                print(f"✅ Loaded {len(all_subs)} subtitles")
            else:
                print("❌ Subtitle file (test1.srt) not found!")
                return False

            # Extract story for key moments
            try:
                from backend.full_story_extractor import FullStoryExtractor
                extractor = FullStoryExtractor()
                sub_list = [{'index': s.index, 'text': s.content, 'start': s.start.total_seconds(), 'end': s.end.total_seconds()} for s in all_subs]
                os.makedirs('temp', exist_ok=True)
                with open('temp/all_subs.json', 'w') as f: json.dump(sub_list, f)
                
                story_subs = extractor.extract_full_story('temp/all_subs.json')
                story_indices = {s.get('index') for s in story_subs}
                filtered_subs = [sub for sub in all_subs if sub.index in story_indices]
                print(f"📚 Full story: {len(filtered_subs)} key moments from {len(all_subs)} total")
            except Exception as e:
                print(f"⚠️ Full story extraction failed, using all subtitles: {e}")
                filtered_subs = all_subs
            
            # Convert to key moments format
            key_moments = [{'index': s.index, 'text': s.content, 'start': s.start.total_seconds(), 'end': s.end.total_seconds()} for s in filtered_subs]
            
            # Save key moments for reference
            with open(os.path.join(self.output_dir, 'key_moments.json'), 'w', encoding='utf-8') as f:
                json.dump(key_moments, f, indent=2)
            
            # Generate frames at key moments
            print("🎬 Extracting frames at key moments...")
            if not self.generate_keyframes_from_moments(self.video_path, key_moments, max_frames=48):
                print("❌ Keyframe extraction failed.")
                return False

            print("✂️ Cropping black bars...")
            black_x, black_y, _, _ = black_bar_crop()
            print("✅ Black bars cropped.")

            print("🎨 Enhancing images...")
            self._enhance_all_images()
            self._enhance_quality_colors()
            print("✅ Images enhanced.")

            print("💬 Creating AI bubbles with key moment dialogues...")
            bubbles = self._create_ai_bubbles_from_moments(black_x, black_y)
            print(f"✅ Created {len(bubbles)} bubbles.")

            print("📋 Generating pages...")
            pages = self._generate_pages(bubbles)
            print(f"✅ Generated {len(pages)} pages.")

            print("💾 Saving results...")
            self._save_results(pages)
            print("✅ Results saved.")

            execution_time = (time.time() - start_time) / 60
            print(f"✅ Comic generation completed in {execution_time:.2f} minutes")
            return True
        except Exception as e:
            print(f"❌ Comic generation failed: {e}")
            traceback.print_exc()
            return False

    def _enhance_all_images(self, single_image_path=None):
        """Enhances colors for a batch of images."""
        target_dir = self.frames_dir
        if single_image_path:
            target_dir = os.path.dirname(single_image_path)
        if not os.path.exists(target_dir): return
        try:
            enhancer = SimpleColorEnhancer()
            enhancer.enhance_batch(target_dir)
        except Exception as e:
            print(f"❌ Simple enhancement failed: {e}")

    def _enhance_quality_colors(self, single_image_path=None):
        """Enhances quality and colors for a batch of images."""
        target_dir = self.frames_dir
        if single_image_path:
            target_dir = os.path.dirname(single_image_path)
        try:
            enhancer = QualityColorEnhancer()
            enhancer.batch_enhance(target_dir)
        except Exception as e:
            print(f"⚠️ Quality enhancement failed: {e}")

    def _create_ai_bubbles_from_moments(self, black_x, black_y):
        """Create bubbles using the key moments dialogues"""
        bubbles = []
        frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        
        # Load frame metadata with dialogues
        metadata_path = 'frames/frame_metadata.json'
        if not os.path.exists(metadata_path):
            print("⚠️ Frame metadata not found, using empty bubbles")
            return [bubble(bubble_offset_x=50, bubble_offset_y=20, lip_x=-1, lip_y=-1, dialog="", emotion='normal') for _ in frame_files]
        
        with open(metadata_path, 'r') as f:
            frame_metadata = json.load(f)
        
        for frame_file in frame_files:
            frame_path = os.path.join(self.frames_dir, frame_file)
            dialogue = ""
            
            # Get dialogue from metadata
            if frame_file in frame_metadata:
                dialogue = frame_metadata[frame_file]['dialogue']
            
            try:
                lip_x, lip_y = -1, -1
                faces = face_detector.detect_faces(frame_path)
                if faces:
                    lip_x, lip_y = face_detector.get_lip_position(frame_path, faces[0])
                
                bubble_x, bubble_y = ai_bubble_placer.place_bubble_ai(frame_path, (lip_x, lip_y))
                bubbles.append(bubble(
                    bubble_offset_x=bubble_x, bubble_offset_y=bubble_y,
                    lip_x=lip_x, lip_y=lip_y, dialog=dialogue, emotion='normal'
                ))
            except Exception:
                bubbles.append(bubble(
                    bubble_offset_x=50, bubble_offset_y=20,
                    lip_x=-1, lip_y=-1, dialog=dialogue, emotion='normal'
                ))
        return bubbles

    def _generate_pages(self, bubbles):
        try:
            from backend.fixed_12_pages_800x1080 import generate_12_pages_800x1080
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            return generate_12_pages_800x1080(frame_files, bubbles)
        except ImportError:
            pages = []
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            frames_per_page = 4
            num_pages = (len(frame_files) + frames_per_page - 1) // frames_per_page
            frame_counter = 0
            for i in range(num_pages):
                page_panels, page_bubbles = [], []
                for _ in range(frames_per_page):
                    if frame_counter < len(frame_files):
                        page_panels.append(panel(
                            image=frame_files[frame_counter], row_span=6, col_span=6
                        ))
                        page_bubbles.append(bubbles[frame_counter] if frame_counter < len(bubbles) else bubble(dialog=""))
                        frame_counter += 1
                if page_panels:
                    pages.append(Page(panels=page_panels, bubbles=page_bubbles))
            return pages

    def _save_results(self, pages):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            pages_data = []
            for page in pages:
                page_dict = {
                    'panels': [p if isinstance(p, dict) else p.__dict__ for p in page.panels],
                    'bubbles': [b if isinstance(b, dict) else b.__dict__ for b in page.bubbles]
                }
                pages_data.append(page_dict)
            with open(os.path.join(self.output_dir, 'pages.json'), 'w', encoding='utf-8') as f:
                json.dump(pages_data, f, indent=2)
            self._copy_template_files()
            print("✅ Results saved successfully!")
        except Exception as e:
            print(f"Save results failed: {e}")
            traceback.print_exc()

    def _copy_template_files(self):
        """This function now includes the working 'Replace Image', 'Flip Bubble', and Panel Gaps features."""
        try:
            template_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Comic - Interactive Editor</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <style>
        body { margin: 0; padding: 20px; background: #f0f0f0; font-family: Arial, sans-serif; }
        .comic-container { max-width: 1200px; margin: 0 auto; }
        .comic-page {
            background: white; width: 600px; height: 400px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1); box-sizing: content-box;
            position: relative; overflow: hidden; border: 1px solid #333;
            padding: 10px;
        }
        .comic-grid {
            display: grid;
            grid-template-columns: 285px 285px;
            grid-template-rows: 185px 185px;
            gap: 10px;
            width: 100%; height: 100%;
        }
        .page-wrapper { margin: 30px auto; width: 622px; display: flex; flex-direction: column; align-items: center; }
        .page-title { text-align: center; color: #333; margin-bottom: 10px; font-size: 18px; font-weight: bold; }
        .panel {
            position: relative; overflow: hidden; width: 100%; height: 100%;
            box-sizing: border-box; cursor: pointer; border: 1px solid #333;
        }
        .panel.selected { outline: 3px solid #2196F3; outline-offset: -3px; }
        .panel img { width: 100%; height: 100%; object-fit: cover; object-position: center; }
        .speech-bubble {
            position: absolute; display: flex; justify-content: center; align-items: center;
            width: auto; height: auto;
            min-width: 50px; max-width: 220px; min-height: 30px; 
            box-sizing: border-box; padding: 8px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3); z-index: 10;
            cursor: move; overflow: visible; font-size: 13px; font-weight: bold; text-align: center;
        }
        .bubble-text { padding: 2px; word-wrap: break-word; }
        .speech-bubble.selected { outline: 2px dashed #4CAF50; }
        .speech-bubble textarea {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; box-sizing: border-box;
            border: 1px solid #4CAF50; background: rgba(255,255,255,0.95);
            font: inherit; text-align: center; resize: none; padding: 8px; z-index: 102;
        }
        /* --- Bubble Styles --- */
        .speech-bubble.speech { background: white; border: 2px solid #333; color: #333; border-radius: 15px; }
        .speech-bubble.thought { background: white; border: 2px dashed #555; color: #333; border-radius: 50%; }
        .speech-bubble.reaction { background: #FFD700; border: 3px solid #E53935; color: #D32F2F; font-weight: 900; text-transform: uppercase; width: 180px; clip-path: polygon(0% 25%, 17% 21%, 17% 0%, 31% 16%, 50% 4%, 69% 16%, 83% 0%, 83% 21%, 100% 25%, 85% 45%, 95% 62%, 82% 79%, 100% 97%, 79% 89%, 60% 98%, 46% 82%, 27% 95%, 15% 78%, 5% 62%, 15% 45%); }
        .speech-bubble.narration { background: #FAFAFA; border: 2px solid #BDBDBD; color: #424242; border-radius: 3px; }
        .speech-bubble.idea { background: linear-gradient(180deg,#FFFDD0 0%, #FFF8B5 100%); border: 2px solid #FFA500; color: #6a4b00; border-radius: 40% 60% 40% 60% / 60% 40% 60% 40%; }
        /* --- Tail and Dot Styles (4-Direction Flip) --- */
        .speech-bubble.speech::after, .speech-bubble.idea::after { content: ''; position: absolute; width: 0; height: 0; border-left: 10px solid transparent; border-right: 10px solid transparent; }
        .speech-bubble.speech::after { border-top: 10px solid #333; bottom: -9px; left: 20px; }
        .speech-bubble.idea::after { border-top: 10px solid #FFA500; bottom: -9px; left: 20px; }
        .speech-bubble.thought::after { display: none; }
        .thought-dot { position: absolute; background-color: white; border: 2px solid #555; border-radius: 50%; z-index: -1; }
        .thought-dot-1 { width: 20px; height: 20px; bottom: -20px; left: 15px; }
        .thought-dot-2 { width: 12px; height: 12px; bottom: -32px; left: 5px; }
        /* Horizontal Flip */
        .speech-bubble.flipped.speech::after, .speech-bubble.flipped.idea::after { left: auto; right: 20px; }
        .speech-bubble.flipped.thought .thought-dot-1 { left: auto; right: 15px; }
        .speech-bubble.flipped.thought .thought-dot-2 { left: auto; right: 5px; }
        /* Vertical Flip */
        .speech-bubble.flipped-vertical.speech::after, .speech-bubble.flipped-vertical.idea::after { bottom: auto; top: -9px; transform: rotate(180deg); }
        .speech-bubble.flipped-vertical.thought .thought-dot-1 { bottom: auto; top: -20px; }
        .speech-bubble.flipped-vertical.thought .thought-dot-2 { bottom: auto; top: -32px; }
        .edit-controls {
            position: fixed; bottom: 20px; right: 20px; background: rgba(44, 62, 80, 0.9);
            color: white; padding: 10px 15px; border-radius: 8px; font-size: 13px;
            z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.3); width: 220px;
        }
        .edit-controls h4 { margin: 0 0 10px 0; color: #26a69a; text-align: center; }
        .edit-controls button, .edit-controls select { margin-top: 5px; padding: 6px 8px; font-size: 12px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; width: 100%; box-sizing: border-box; }
        .edit-controls .control-group { margin-top: 10px; border-top: 1px solid #555; padding-top: 10px; }
        .edit-controls .reset-button { background-color: #e74c3c; }
        .edit-controls .action-button { background-color: #4CAF50; }
        .edit-controls .secondary-button { background-color: #f39c12; }
    </style>
</head>
<body>
    <div class="comic-container">
        <h1 class="comic-title">🎬 Generated Comic</h1>
        <div id="comic-pages"><div class="loading">Loading comic...</div></div>
    </div>
    <input type="file" id="image-uploader" style="display: none;" accept="image/*">
    
    <div class="edit-controls">
        <h4>✏️ Interactive Editor</h4>
        <div class="control-group">
            <label for="bubble-type-select">Change Selected Bubble Type:</label>
            <select id="bubble-type-select" onchange="changeBubbleType(this.value)">
                <option value="speech">Speech</option>
                <option value="thought">Thought</option>
                <option value="reaction">Reaction</option>
                <option value="narration">Narration</option>
                <option value="idea">Idea</option>
            </select>
            <button onclick="rotateBubbleTail()" class="secondary-button">🔄 Rotate Tail</button>
        </div>
        <div class="control-group">
             <button onclick="replacePanelImage()" class="action-button">🖼️ Replace Panel Image</button>
             <button onclick="regenerateFrame()" class="action-button">🔄 Regenerate Frame</button>
             <button onclick="exportPagesToPNG()" class="action-button" style="background-color: #2196F3;">🖨️ Export Pages to PNG</button>
        </div>
        <div class="control-group">
             <button onclick="clearSavedState()" class="reset-button">🔄 Clear Edits & Reset</button>
        </div>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            fetch('/output/pages.json')
                .then(res => res.ok ? res.json() : Promise.reject(new Error('Failed to load pages.json')))
                .then(data => { renderComic(data); initializeEditor(); })
                .catch(err => { document.getElementById('comic-pages').innerHTML = `<div class="loading">Error: ${err.message}</div>`; });
        });
        
        function renderComic(data) {
            const container = document.getElementById('comic-pages');
            container.innerHTML = '';
            if (!data || data.length === 0) return;
            data.forEach((pageData, pageIndex) => {
                if (!pageData.panels || pageData.panels.length === 0) return;
                const pageWrapper = document.createElement('div');
                pageWrapper.className = 'page-wrapper';
                const pageTitleEl = document.createElement('h2');
                pageTitleEl.className = 'page-title';
                pageTitleEl.textContent = `Page ${pageIndex + 1}`;
                pageWrapper.appendChild(pageTitleEl);
                const pageDiv = document.createElement('div');
                pageDiv.className = 'comic-page';
                const grid = document.createElement('div');
                grid.className = 'comic-grid';
                pageData.panels.forEach((panelData, panelIndex) => {
                    const panelDiv = document.createElement('div');
                    panelDiv.className = 'panel';
                    const img = document.createElement('img');
                    img.src = '/frames/final/' + panelData.image;
                    panelDiv.appendChild(img);
                    if (pageData.bubbles && pageData.bubbles[panelIndex]) {
                        const bubbleData = pageData.bubbles[panelIndex];
                        const bubbleDiv = createBubbleElement({
                            id: `initial-${pageIndex}-${panelIndex}`,
                            text: bubbleData.dialog || '',
                            left: `${bubbleData.bubble_offset_x ?? 50}px`,
                            top: `${bubbleData.bubble_offset_y ?? 20}px`,
                        });
                        panelDiv.appendChild(bubbleDiv);
                    }
                    grid.appendChild(panelDiv);
                });
                pageDiv.appendChild(grid);
                pageWrapper.appendChild(pageDiv);
                container.appendChild(pageWrapper);
            });
        }
        
        let currentlyEditing = null, draggedBubble = null, offset = {x: 0, y: 0};
        let currentlySelectedBubble = null;
        let currentlySelectedPanel = null;
        
        function initializeEditor() {
            document.querySelectorAll('.panel').forEach(p => p.addEventListener('click', e => selectPanel(e.currentTarget)));
            document.querySelectorAll('.speech-bubble').forEach(b => initializeBubbleEvents(b));
            document.addEventListener('mousemove', e => { if (draggedBubble) drag(e); });
            document.addEventListener('mouseup', () => { if (draggedBubble) stopDrag(); });
        }
        
        function initializeBubbleEvents(bubble) {
            bubble.addEventListener('dblclick', e => { e.stopPropagation(); editBubbleText(bubble); });
            bubble.addEventListener('mousedown', e => startDrag(e));
            bubble.addEventListener('click', e => { e.stopPropagation(); selectBubble(bubble); });
            bubble.addEventListener('wheel', e => {
                e.preventDefault();
                const currentWidth = parseFloat(bubble.style.width) || bubble.offsetWidth;
                const newWidth = currentWidth - (e.deltaY > 0 ? 10 : -10);
                if (newWidth >= 60) {
                    bubble.style.width = `${newWidth}px`;
                    bubble.style.height = 'auto';
                }
            }, { passive: false });
        }
        
        function createBubbleElement(data) {
            const bubbleDiv = document.createElement('div');
            bubbleDiv.dataset.id = data.id;
            const textSpan = document.createElement('span');
            textSpan.className = 'bubble-text';
            textSpan.textContent = data.text;
            bubbleDiv.appendChild(textSpan);
            bubbleDiv.style.left = data.left;
            bubbleDiv.style.top = data.top;
            applyBubbleType(bubbleDiv, 'speech'); // Default to speech
            return bubbleDiv;
        }
        
        function applyBubbleType(bubble, type) {
            bubble.querySelectorAll('.thought-dot').forEach(el => el.remove());
            let classesToKeep = 'speech-bubble';
            if (bubble.classList.contains('selected')) classesToKeep += ' selected';
            if (bubble.classList.contains('flipped')) classesToKeep += ' flipped';
            if (bubble.classList.contains('flipped-vertical')) classesToKeep += ' flipped-vertical';
            bubble.className = classesToKeep;
            bubble.classList.add(type);
            bubble.dataset.type = type;
            if (type === 'thought') {
                for (let i = 1; i <= 2; i++) {
                    const dot = document.createElement('div');
                    dot.className = `thought-dot thought-dot-${i}`;
                    bubble.appendChild(dot);
                }
            }
        }
        
        function changeBubbleType(type) {
            if (!currentlySelectedBubble) return;
            applyBubbleType(currentlySelectedBubble, type);
        }
        
        function rotateBubbleTail() {
            if (!currentlySelectedBubble) return alert("Please select a bubble to rotate.");
            const isFlippedH = currentlySelectedBubble.classList.contains('flipped');
            const isFlippedV = currentlySelectedBubble.classList.contains('flipped-vertical');
            if (!isFlippedH && !isFlippedV) { // State 0 -> 1
                currentlySelectedBubble.classList.add('flipped');
            } else if (isFlippedH && !isFlippedV) { // State 1 -> 2
                currentlySelectedBubble.classList.add('flipped-vertical');
            } else if (isFlippedH && isFlippedV) { // State 2 -> 3
                currentlySelectedBubble.classList.remove('flipped');
            } else { // State 3 -> 0
                currentlySelectedBubble.classList.remove('flipped-vertical');
            }
        }
        
        function selectPanel(panel) {
            document.querySelectorAll('.panel.selected').forEach(p => p.classList.remove('selected'));
            panel.classList.add('selected');
            currentlySelectedPanel = panel;
            selectBubble(null);
        }
        
        function selectBubble(bubble) {
            if (currentlySelectedBubble) currentlySelectedBubble.classList.remove('selected');
            currentlySelectedBubble = bubble;
            if (currentlySelectedBubble) {
                currentlySelectedBubble.classList.add('selected');
                document.querySelectorAll('.panel.selected').forEach(p => p.classList.remove('selected'));
                document.getElementById('bubble-type-select').value = currentlySelectedBubble.dataset.type || 'speech';
            }
        }
        
        function editBubbleText(bubble) {
            if (currentlyEditing) return;
            currentlyEditing = bubble;
            const textSpan = bubble.querySelector('.bubble-text');
            const currentText = textSpan.textContent;
            textSpan.style.display = 'none';
            bubble.style.height = 'auto';
            const textarea = document.createElement('textarea');
            textarea.value = currentText;
            bubble.appendChild(textarea);
            textarea.focus();
            const finishEditing = () => {
                textSpan.textContent = textarea.value;
                bubble.removeChild(textarea);
                textSpan.style.display = '';
                currentlyEditing = null;
                bubble.style.height = 'auto';
            };
            textarea.addEventListener('blur', finishEditing, { once: true });
            textarea.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); textarea.blur(); }});
        }
        
        function startDrag(e) {
            const bubble = e.target.closest('.speech-bubble');
            if (!bubble || currentlyEditing) return;
            draggedBubble = bubble;
            selectBubble(bubble);
            const rect = bubble.getBoundingClientRect();
            offset = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        }
        
        function drag(e) {
            const parentRect = draggedBubble.parentElement.getBoundingClientRect();
            let x = e.clientX - parentRect.left - offset.x;
            let y = e.clientY - parentRect.top - offset.y;
            draggedBubble.style.left = `${x}px`;
            draggedBubble.style.top = `${y}px`;
        }
        
        function stopDrag() {
            draggedBubble = null;
        }
        
        function clearSavedState() {
            if (confirm("Reset all edits to the original AI-generated comic?")) {
                localStorage.removeItem('comicEditorState');
                window.location.reload();
            }
        }
        
        async function exportPagesToPNG() {
            const pages = document.querySelectorAll('.comic-page');
            if (pages.length === 0) return alert("No pages found.");
            alert(`Starting export of ${pages.length} page(s).`);
            for (let i = 0; i < pages.length; i++) {
                try {
                    const canvas = await html2canvas(pages[i], { scale: 2 });
                    const link = document.createElement('a');
                    link.download = `comic-page-${i + 1}.png`;
                    link.href = canvas.toDataURL('image/png');
                    link.click();
                } catch (err) {
                    alert(`Failed to export page ${i + 1}.`);
                }
            }
        }
        
        function replacePanelImage() {
            if (!currentlySelectedPanel) {
                alert("Please select a panel first.");
                return;
            }
            const img = currentlySelectedPanel.querySelector('img');
            const uploader = document.getElementById('image-uploader');
            const oneTimeListener = (event) => {
                const file = event.target.files[0];
                if (!file) return;
                const formData = new FormData();
                formData.append('image', file);
                img.style.opacity = '0.5';
                fetch('/replace_panel', { method: 'POST', body: formData })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            img.src = `/frames/final/${data.new_filename}?t=${new Date().getTime()}`;
                        } else {
                            alert('Error replacing image: ' + data.error);
                        }
                        img.style.opacity = '1';
                    })
                    .catch(error => {
                        alert('An error occurred during the upload.');
                        img.style.opacity = '1';
                    });
                uploader.removeEventListener('change', oneTimeListener);
                uploader.value = '';
            };
            uploader.addEventListener('change', oneTimeListener, { once: true });
            uploader.click();
        }
        
        function regenerateFrame() {
            if (!currentlySelectedPanel) {
                alert("Please select a panel first.");
                return;
            }
            const img = currentlySelectedPanel.querySelector('img');
            const currentSrc = img.src;
            
            let filename = currentSrc.substring(currentSrc.lastIndexOf('/') + 1);
            if (filename.includes('?')) {
                filename = filename.split('?')[0];
            }

            if (!confirm(`Regenerate frame "${filename}" with a better version?`)) {
                return;
            }
            img.style.opacity = '0.5';
            fetch('/regenerate_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: filename })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    img.src = `/frames/final/${filename}?t=${new Date().getTime()}`;
                    alert(data.message);
                } else {
                    alert('Error: ' + data.message);
                }
                img.style.opacity = '1';
            })
            .catch(error => {
                alert('An error occurred during regeneration.');
                img.style.opacity = '1';
            });
        }
    </script>
</body>
</html>'''
            with open(os.path.join(self.output_dir, 'page.html'), 'w', encoding='utf-8') as f:
                f.write(template_html)
            print("📄 Template files copied successfully!")
        except Exception as e:
            print(f"Template copy failed: {e}")

# Flask routes
comic_generator = EnhancedComicGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploader', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return "❌ No file selected"
        f = request.files['file']
        if os.path.exists(comic_generator.video_path):
             os.remove(comic_generator.video_path)
        f.save(comic_generator.video_path)
        success = comic_generator.generate_comic()
        if success:
            webbrowser.open("http://localhost:5000/comic")
            return "🎉 Enhanced Comic Created Successfully!"
        else:
            return "❌ Comic generation failed"
    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.route('/handle_link', methods=['POST'])
def handle_link():
    try:
        link = request.form.get('link', '')
        if not link:
            return "❌ No link provided"
        import yt_dlp
        ydl_opts = {'outtmpl': comic_generator.video_path, 'format': 'best[height<=720]'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
        success = comic_generator.generate_comic()
        if success:
            webbrowser.open("http://localhost:5000/comic")
            return "🎉 Enhanced Comic Created Successfully!"
        else:
            return "❌ Comic generation failed"
    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.route('/replace_panel', methods=['POST'])
def replace_panel():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided.'})
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected.'})
        timestamp = int(time.time() * 1000)
        filename = f"replaced_panel_{timestamp}.png"
        save_path = os.path.join(comic_generator.frames_dir, filename)
        file.save(save_path)
        
        # --- FIX: Color enhancement is now skipped for replaced images ---
        # print(f"🖼️ Enhancing replaced panel image: {filename}")
        # comic_generator._enhance_all_images(single_image_path=save_path)
        # comic_generator._enhance_quality_colors(single_image_path=save_path)
        # print(f"✅ Enhancement complete for {filename}")
        print(f"✅ Replaced panel with '{filename}' without applying color enhancement.")
        
        return jsonify({'success': True, 'new_filename': filename})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/regenerate_frame', methods=['POST'])
def regenerate_frame_route():
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({'success': False, 'message': 'No filename provided'})
        result = comic_generator.regenerate_frame(filename)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/comic')
def view_comic():
    return send_from_directory('output', 'page.html')

@app.route('/output/<path:filename>')
def output_file(filename):
    return send_from_directory('output', filename)

@app.route('/frames/final/<path:filename>')
def frame_file(filename):
    return send_from_directory('frames/final', filename)

if __name__ == '__main__':
    print("🚀 Starting Enhanced Comic Generator...")
    print("🌐 Web interface available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
