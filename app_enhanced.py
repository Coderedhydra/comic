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
from typing import List

# Import enhanced modules
from backend.ai_enhanced_core import (
    image_processor, comic_styler, face_detector, layout_optimizer
)
from backend.ai_bubble_placement import ai_bubble_placer
from backend.subtitles.subs_real import get_real_subtitles
from backend.keyframes.keyframes_simple import generate_keyframes_simple
from backend.keyframes.keyframes import black_bar_crop
from backend.class_def import bubble, panel, Page

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
        self.quality_mode = os.getenv('HIGH_QUALITY', '1')
        self.ai_mode = os.getenv('AI_ENHANCED', '1')
        self.apply_comic_style = False  # Disabled to preserve original colors
        self.preserve_colors = True  # Preserve more original colors in comic style
        
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
        
    def generate_comic(self, smart_mode=False, emotion_match=False):
        """Main comic generation pipeline
        
        Args:
            smart_mode: If True, generates 10-15 panel summary
            emotion_match: If True, matches facial expressions with dialogue
        """
        start_time = time.time()
        
        print("🎬 Starting Enhanced Comic Generation...")
        if smart_mode:
            print("🎭 Smart mode enabled: Will create 10-15 panel summary with emotion matching")
        
        try:
            # 1. Extract real subtitles from video audio
            print("📝 Extracting real subtitles from video...")
            get_real_subtitles(self.video_path)
            
            # 2. Extract FULL story (don't skip important parts)
            print("📖 Extracting complete story...")
            filtered_subs = None
            if os.path.exists('test1.srt'):
                try:
                    from backend.full_story_extractor import FullStoryExtractor
                    extractor = FullStoryExtractor()
                    
                    # Get all subtitles first
                    with open('test1.srt', 'r', encoding='utf-8') as f:
                        all_subs = list(srt.parse(f.read()))
                    
                    # Convert to dict format
                    sub_list = []
                    for sub in all_subs:
                        sub_list.append({
                            'index': sub.index,
                            'text': sub.content,
                            'start': sub.start.total_seconds(),
                            'end': sub.end.total_seconds()
                        })
                    
                    # Save temp file
                    os.makedirs('temp', exist_ok=True)
                    with open('temp/all_subs.json', 'w') as f:
                        json.dump(sub_list, f)
                    
                    # Extract full story (up to 48 panels)
                    story_subs = extractor.extract_full_story('temp/all_subs.json')
                    
                    # Convert back to srt format
                    filtered_subs = []
                    for s in story_subs:
                        # Find matching subtitle
                        for sub in all_subs:
                            if sub.index == s.get('index', -1):
                                filtered_subs.append(sub)
                                break
                    
                    print(f"📚 Full story: {len(filtered_subs)} key moments from {len(all_subs)} total")
                    
                except Exception as e:
                    print(f"⚠️ Full story extraction failed: {e}")
                    filtered_subs = None
            
            # 3. Generate keyframes based on story moments
            print("🎯 Generating keyframes...")
            if filtered_subs:
                # Use story-based keyframe generation
                from backend.keyframes.keyframes_story import generate_keyframes_story
                generate_keyframes_story(self.video_path, filtered_subs)
            else:
                # Fallback to simple method
                generate_keyframes_simple(self.video_path)
            
            # 4. Remove black bars
            print("✂️ Removing black bars...")
            black_x, black_y, _, _ = black_bar_crop()
            
            # 5. Enhance image quality with advanced models
            if self.quality_mode == '1':
                print("✨ Enhancing image quality with advanced AI models...")
                self._enhance_all_images_advanced()
            
            # 6. Apply quality and color enhancement
            print("🎨 Enhancing quality and colors...")
            self._enhance_quality_colors()
            
            # 7. Apply comic styling (if enabled)
            print("🎨 Applying AI-enhanced comic styling...")
            self._apply_comic_styling()
            
            # 7. Generate optimized layout
            print("📐 Generating AI-optimized layout...")
            layout_data = self._generate_optimized_layout()
            
            # 8. Create AI-powered speech bubbles
            print("💬 Creating AI-powered speech bubbles...")
            bubbles = self._create_ai_bubbles(black_x, black_y)
            
            # 9. Generate final pages
            print("📄 Generating final pages...")
            pages = self._generate_pages(layout_data, bubbles)
            
            # 10. Save results
            print("💾 Saving results...")
            self._save_results(pages)
            
            # 11. Generate smart comic if requested
            if smart_mode and SMART_COMIC_AVAILABLE:
                print("\n🎭 Generating smart comic with emotion matching...")
                self._generate_smart_comic(emotion_match)
            
            # 12. Extract individual panels as 640x800 images
            print("\n📸 Extracting individual panels...")
            self._extract_panels()
            
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
    
    def _enhance_quality_colors(self):
        """Enhance image quality and colors"""
        try:
            from backend.quality_color_enhancer import QualityColorEnhancer
            enhancer = QualityColorEnhancer()
            enhancer.batch_enhance(self.frames_dir)
        except Exception as e:
            print(f"⚠️ Quality enhancement failed: {e}")
    
    def _apply_comic_styling(self):
        """Apply comic styling to all frames"""
        if not self.apply_comic_style:
            print("⏭️ Skipping comic styling to preserve original colors")
            return
            
        if not os.path.exists(self.frames_dir):
            print(f"❌ Frames directory not found: {self.frames_dir}")
            return
            
        frame_files = [f for f in os.listdir(self.frames_dir) if f.endswith('.png')]
        print(f"Found {len(frame_files)} frames to style")
        
        # Set color preservation mode
        if hasattr(comic_styler, 'preserve_colors'):
            comic_styler.preserve_colors = self.preserve_colors
            print(f"🎨 Comic styling with color preservation: {self.preserve_colors}")
        
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
            # Count frames
            frame_count = len([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            
            # If we have filtered story moments, use adaptive layout
            if STORY_EXTRACTOR_AVAILABLE and hasattr(self, '_filtered_count'):
                extractor = SmartStoryExtractor()
                layouts = extractor.get_adaptive_layout(self._filtered_count)
                
                # Generate layout strings based on adaptive layout
                layout_strings = []
                panel_idx = 0
                
                for page_layout in layouts:
                    panels_on_page = page_layout['panels_per_page']
                    rows = page_layout['rows']
                    cols = page_layout['cols']
                    
                    # Create rows for this page
                    for row in range(rows):
                        row_string = ""
                        for col in range(cols):
                            if panel_idx < self._filtered_count:
                                row_string += str(panel_idx % 10)  # Use single digit
                                panel_idx += 1
                            else:
                                row_string += "0"  # Empty panel
                        layout_strings.append(row_string)
                
                print(f"✅ Generated adaptive layout for {self._filtered_count} story panels")
                return layout_strings
            else:
                # Use default layout optimizer with frame paths
                frame_paths = [os.path.join(self.frames_dir, f) for f in os.listdir(self.frames_dir) if f.endswith('.png')]
                layout_data = layout_optimizer.optimize_layout(frame_paths)
                return layout_data
                
        except Exception as e:
            print(f"Layout generation failed: {e}")
            # Fallback to simple 2x2 layout
            return ['6666', '6666', '6666', '6666']
    
    def _filter_meaningful_subtitles(self, srt_path: str) -> List:
        """Filter subtitles to keep only meaningful story moments"""
        if not STORY_EXTRACTOR_AVAILABLE:
            return None
            
        try:
            # Read all subtitles
            with open(srt_path, 'r', encoding='utf-8') as f:
                all_subs = list(srt.parse(f.read()))
            
            # Convert to JSON format for extractor
            sub_json = []
            for sub in all_subs:
                sub_json.append({
                    'text': sub.content,
                    'start': str(sub.start),
                    'end': str(sub.end),
                    'index': sub.index
                })
            
            # Save temporarily
            os.makedirs('audio', exist_ok=True)
            temp_json = 'audio/temp_subtitles.json'
            with open(temp_json, 'w') as f:
                json.dump(sub_json, f)
            
            # Extract meaningful moments
            extractor = SmartStoryExtractor()
            meaningful = extractor.extract_meaningful_story(temp_json, target_panels=12)
            
            # Convert back to SRT objects
            meaningful_indices = [m['index'] for m in meaningful]
            filtered_subs = [sub for sub in all_subs if sub.index in meaningful_indices]
            
            print(f"📖 Filtered {len(all_subs)} subtitles to {len(filtered_subs)} key moments")
            
            # Store filtered count for layout generation
            self._filtered_count = len(filtered_subs)
            
            return filtered_subs
            
        except Exception as e:
            print(f"⚠️ Subtitle filtering failed: {e}")
            return None
    
    def _create_ai_bubbles(self, black_x, black_y):
        """Create AI-powered speech bubbles"""
        bubbles = []
        
        try:
            # Read and filter subtitles
            srt_path = 'test1.srt'
            
            # Try to filter for meaningful moments
            filtered_subs = self._filter_meaningful_subtitles(srt_path)
            
            if filtered_subs:
                subs = filtered_subs
            else:
                # Fallback to all subtitles
                with open(srt_path, 'r', encoding='utf-8') as f:
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
        """Generate final pages based on story-aware layout"""
        pages = []
        
        try:
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            
            # Always use story-based layout for proper grid
            # Force 12 panels for meaningful story
            if not hasattr(self, '_filtered_count'):
                self._filtered_count = min(12, len(frame_files))
            return self._generate_story_pages(frame_files, bubbles)
            
            # Otherwise, create simple layout based on available frames
            num_frames = len(frame_files)
            frames_per_page = 6 if num_frames > 9 else (4 if num_frames > 4 else num_frames)
            num_pages = (num_frames + frames_per_page - 1) // frames_per_page
            
            frame_idx = 0
            for page_num in range(num_pages):
                panels = []
                page_bubbles = []
                
                # Force proper grid layout for 12 panels
                if num_frames <= 6:
                    rows, cols = 2, 3
                elif num_frames <= 9:
                    rows, cols = 3, 3
                elif num_frames <= 12:
                    rows, cols = 3, 4
                else:
                    rows, cols = 4, 4
                
                for j in range(frames_per_page):
                    if frame_idx < len(frame_files):
                        frame_file = frame_files[frame_idx]
                    
                    # Calculate proper span for grid
                    row_span = 12 // rows
                    col_span = 12 // cols
                    
                    panel_obj = panel(
                        image=frame_file,
                        row_span=row_span,
                        col_span=col_span
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
    
    def _generate_story_pages(self, frame_files, bubbles):
        """Generate pages based on story extraction"""
        # Use 2x2 grid with 12 PAGES (48 panels total)
        from backend.fixed_12_pages_2x2 import generate_12_pages_2x2_grid
        
        print(f"📖 Generating 12-page comic summary (2x2 grid per page)")
        print(f"📊 Target: 48 meaningful panels from {len(frame_files)} frames")
        
        return generate_12_pages_2x2_grid(frame_files, bubbles)
        
        # Get adaptive layout configuration
        if STORY_EXTRACTOR_AVAILABLE:
            extractor = SmartStoryExtractor()
            layouts = extractor.get_adaptive_layout(len(frame_files))
        else:
            # Fallback layout
            layouts = [{'panels_per_page': 6, 'rows': 2, 'cols': 3}]
        
        frame_idx = 0
        bubble_idx = 0
        
        for page_num, layout_config in enumerate(layouts):
            panels = []
            page_bubbles = []
            
            panels_on_page = layout_config['panels_per_page']
            rows = layout_config['rows']
            cols = layout_config['cols']
            
            # Calculate panel dimensions
            row_span = 12 // rows
            col_span = 12 // cols
            
            for panel_num in range(panels_on_page):
                if frame_idx < len(frame_files):
                    # Create panel
                    panel_obj = panel(
                        image=frame_files[frame_idx],
                        row_span=row_span,
                        col_span=col_span
                    )
                    panels.append(panel_obj)
                    
                    # Add corresponding bubble if available
                    if bubble_idx < len(bubbles):
                        bubble_obj = bubbles[bubble_idx]
                        page_bubbles.append(bubble_obj)
                        bubble_idx += 1
                    
                    frame_idx += 1
                
                if frame_idx >= len(frame_files):
                    break
            
            # Create page
            page = Page(
                panels=panels,
                bubbles=page_bubbles
            )
            pages.append(page)
            
            print(f"📄 Page {page_num + 1}: {len(panels)} panels in {rows}x{cols} grid")
        
        return pages
    
    def _generate_arrangement(self, rows, cols):
        """Generate panel arrangement string for given rows and cols"""
        arrangement = []
        panel_num = 0
        
        for r in range(rows):
            row_str = ""
            for c in range(cols):
                row_str += str(panel_num % 10)
                panel_num += 1
            arrangement.append(row_str)
        
        return arrangement
    
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
    
    def _generate_smart_comic(self, emotion_match=True):
        """Generate smart comic with emotion matching"""
        try:
            if emotion_match:
                # Use emotion-aware generator
                generator = EmotionAwareComicGenerator()
                comic_data = generator.generate_emotion_comic(self.video_path)
            else:
                # Use story analyzer
                generator = SmartComicGenerator()
                comic_data = generator.generate_smart_comic(self.video_path)
            
            # Generate viewer HTML
            if comic_data:
                self._generate_smart_viewer(comic_data)
                print("✅ Smart comic generated: output/smart_comic_viewer.html")
                
        except Exception as e:
            print(f"⚠️ Smart comic generation failed: {e}")
    
    def _generate_smart_viewer(self, comic_data):
        """Generate HTML viewer for smart comic"""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Smart Comic - Emotion Matched</title>
    <style>
        body { margin: 0; padding: 20px; background: #2c3e50; color: white; font-family: Arial, sans-serif; }
        .header { text-align: center; margin-bottom: 30px; }
        .comic-page { position: relative; background: white; margin: 20px auto; box-shadow: 0 10px 40px rgba(0,0,0,0.5); }
        .comic-panel { position: absolute; border: 3px solid #333; overflow: hidden; }
        .comic-panel img { width: 100%; height: 100%; object-fit: cover; }
        .speech-bubble { position: absolute; border-radius: 20px; padding: 12px; font-family: "Comic Sans MS", cursive; font-weight: bold; text-align: center; z-index: 10; }
        .emotion-happy { border: 3px solid #4CAF50; background: #E8F5E9; }
        .emotion-sad { border: 3px solid #2196F3; background: #E3F2FD; }
        .emotion-angry { border: 4px solid #F44336; background: #FFEBEE; }
        .emotion-surprised { border: 3px solid #FF9800; background: #FFF3E0; }
        .emotion-neutral { border: 2px solid #333; background: #FFF; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎭 Smart Comic Summary</h1>
        <p>AI-generated comic with emotion matching (10-15 key panels)</p>
    </div>
'''
        
        for page in comic_data.get('pages', []):
            html += f'<div class="comic-page" style="width:{page["width"]}px;height:{page["height"]}px;margin:20px auto;">\n'
            
            for panel in page.get('panels', []):
                html += f'<div class="comic-panel" style="left:{panel["x"]}px;top:{panel["y"]}px;width:{panel["width"]}px;height:{panel["height"]}px;">'
                html += f'<img src="{panel["image"]}">'
                html += '</div>\n'
            
            for bubble in page.get('bubbles', []):
                emotion = bubble.get('emotion', 'neutral')
                style = bubble.get('style', {})
                html += f'<div class="speech-bubble emotion-{emotion}" style="'
                html += f'left:{bubble["x"]}px;top:{bubble["y"]}px;width:{bubble["width"]}px;min-height:{bubble["height"]}px;">'
                html += bubble["text"]
                html += '</div>\n'
            
            html += '</div>\n'
        
        html += '</body></html>'
        
        with open('output/smart_comic_viewer.html', 'w', encoding='utf-8') as f:
            f.write(html)
    
    def _extract_panels(self):
        """Extract individual panels as 640x800 images"""
        if not PANEL_EXTRACTOR_AVAILABLE:
            print("⚠️ Panel extractor not available, skipping...")
            return
            
        try:
            extractor = PanelExtractor(output_dir="output/panels")
            saved_panels = extractor.extract_panels_from_comic(
                pages_json_path="output/pages.json",
                frames_dir="frames/final"
            )
            
            if saved_panels:
                print(f"✅ Extracted {len(saved_panels)} panels to output/panels/")
                print("📄 Panel viewer available at: output/panels/panel_viewer.html")
            
        except Exception as e:
            print(f"⚠️ Panel extraction failed: {e}")
    
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
            
            # Get smart comic options
            smart_mode = request.form.get('smart_mode', 'false').lower() == 'true'
            emotion_match = request.form.get('emotion_match', 'false').lower() == 'true'
            
            # Generate comic
            success = comic_generator.generate_comic(smart_mode=smart_mode, emotion_match=emotion_match)
            
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
            
            # Get smart comic options
            smart_mode = request.form.get('smart_mode', 'false').lower() == 'true'
            emotion_match = request.form.get('emotion_match', 'false').lower() == 'true'
            
            # Generate comic
            success = comic_generator.generate_comic(smart_mode=smart_mode, emotion_match=emotion_match)
            
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
    # Check if smart comic exists
    smart_comic_path = os.path.join('output', 'smart_comic_viewer.html')
    if os.path.exists(smart_comic_path):
        return send_from_directory('output', 'smart_comic_viewer.html')
    # Otherwise serve regular comic
    return send_from_directory('output', 'page.html')

@app.route('/smart_comic')
def view_smart_comic():
    """Serve the smart comic viewer"""
    return send_from_directory('output', 'smart_comic_viewer.html')

@app.route('/panels')
def view_panels():
    """Serve the panel viewer"""
    return send_from_directory('output/panels', 'panel_viewer.html')

@app.route('/output/panels/<path:filename>')
def panel_file(filename):
    """Serve individual panel files"""
    return send_from_directory('output/panels', filename)

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