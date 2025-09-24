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
    
    def _cleanup_previous_files(self):
        """Clean previous SRT and temporary files to prevent conflicts"""
        try:
            from backend.srt_cleanup import delete_srt_files, delete_temp_files
            
            # Delete SRT files
            srt_result = delete_srt_files(".")
            print(f"   🗑️  Deleted {srt_result['deleted_count']} SRT files")
            
            # Delete temporary files
            temp_result = delete_temp_files(".")
            print(f"   🗑️  Deleted {temp_result['deleted_count']} temporary files")
            
            if srt_result['errors'] or temp_result['errors']:
                print("   ⚠️  Some files could not be deleted (may be in use)")
                
        except Exception as e:
            print(f"   ⚠️  Cleanup warning: {e}")
            # Don't fail the entire process for cleanup issues
    
    def _create_frame_dialogue_sync(self):
        """Create frame-dialogue synchronization mapping"""
        try:
            from backend.frame_dialogue_sync import enhance_bubble_sync
            import srt
            
            # Load subtitles
            if not os.path.exists('test1.srt'):
                print("   ⚠️  No subtitles found for synchronization")
                return
                
            with open('test1.srt', 'r') as f:
                subtitles = list(srt.parse(f.read()))
            
            # Get frame files
            frames_dir = os.path.join(self.frames_dir)
            if not os.path.exists(frames_dir):
                print("   ⚠️  No frames directory found for synchronization")
                return
                
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
            frame_files.sort()
            
            if not frame_files:
                print("   ⚠️  No frame files found for synchronization")
                return
            
            # Create synchronization mapping
            mapping = enhance_bubble_sync(self.video_path, frame_files, subtitles)
            
            print(f"   ✅ Synchronized {len(subtitles)} dialogues with {len(frame_files)} frames")
            
        except Exception as e:
            print(f"   ⚠️  Frame-dialogue sync warning: {e}")
            # Don't fail the entire process for sync issues
        
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
        
        # Clean previous SRT files to prevent conflicts
        print("\n🧹 Cleaning previous SRT files...")
        self._cleanup_previous_files()
        
        try:
            # 1. Extract real subtitles from video audio
            print("📝 Extracting real subtitles from video...")
            get_real_subtitles(self.video_path)
            
            # 1.5. Create frame-dialogue synchronization mapping
            print("🔗 Creating frame-dialogue synchronization...")
            self._create_frame_dialogue_sync()
            
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
                    
                    # Store the count for later use
                    self._filtered_count = len(filtered_subs)
                    
                except Exception as e:
                    print(f"⚠️ Full story extraction failed: {e}")
                    # Create a fallback with limited subtitles instead of None
                    print("🔄 Creating fallback story extraction...")
                    with open('test1.srt', 'r', encoding='utf-8') as f:
                        all_subs = list(srt.parse(f.read()))
                    
                    # Simple fallback: take every Nth subtitle to get ~48
                    if len(all_subs) > 48:
                        step = len(all_subs) // 48
                        filtered_subs = []
                        for i in range(0, len(all_subs), step):
                            if len(filtered_subs) < 48:
                                filtered_subs.append({
                                    'index': all_subs[i].index,
                                    'text': all_subs[i].content,
                                    'start': all_subs[i].start.total_seconds(),
                                    'end': all_subs[i].end.total_seconds()
                                })
                        self._filtered_count = len(filtered_subs)
                        print(f"🔄 Fallback extraction: {len(all_subs)} → {len(filtered_subs)} subtitles")
                    else:
                        # Use all if less than 48
                        filtered_subs = []
                        for sub in all_subs:
                            filtered_subs.append({
                                'index': sub.index,
                                'text': sub.content,
                                'start': sub.start.total_seconds(),
                                'end': sub.end.total_seconds()
                            })
                        self._filtered_count = len(filtered_subs)
            
            # 3. Generate keyframes based on story moments
            print("🎯 Generating keyframes...")
            if filtered_subs and smart_mode:
                # Use ENGAGING frame selection when smart mode is enabled
                print("✨ Selecting most engaging frames...")
                from backend.keyframes.keyframes_engaging import generate_keyframes_engaging
                success = generate_keyframes_engaging(self.video_path, filtered_subs, max_frames=48)
                if not success:
                    print("⚠️ Engaging selection failed, trying smart method...")
                    from backend.keyframes.keyframes_smart import generate_keyframes_smart
                    success = generate_keyframes_smart(self.video_path, filtered_subs, max_frames=48)
                    if not success:
                        print("⚠️ Smart extraction failed, trying fixed method...")
                        from backend.keyframes.keyframes_fixed import generate_keyframes_fixed
                        generate_keyframes_fixed(self.video_path, filtered_subs, max_frames=48)
            elif filtered_subs:
                # Use regular smart extraction (checks eyes but not emotions)
                from backend.keyframes.keyframes_smart import generate_keyframes_smart
                success = generate_keyframes_smart(self.video_path, filtered_subs, max_frames=48)
                if not success:
                    from backend.keyframes.keyframes_fixed import generate_keyframes_fixed
                    generate_keyframes_fixed(self.video_path, filtered_subs, max_frames=48)
            else:
                # Fallback to simple method with improved frame count
                print("🔄 Using simple keyframe method with enhanced frame count...")
                generate_keyframes_simple(self.video_path)
            
            # 4. Remove black bars
            print("✂️ Removing black bars...")
            black_x, black_y, _, _ = black_bar_crop()
            
            # 5. Enhance image quality with advanced models
            if self.quality_mode == '1':
                print("✨ Using simple quality enhancement to avoid color issues...")
                # Skip AI enhancement that causes green tint
                # self._enhance_all_images_advanced()
                
                # Use simple enhancement instead
                self._enhance_all_images()
            
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
            
            # 11. Smart mode already applied during frame selection
            if smart_mode:
                print("✅ Smart frame selection completed")
            
            # 12. Extract individual panels as 640x800 images
            print("\n📸 Extracting individual panels...")
            self._extract_panels()
            
            # 13. Generate page images at 600x400
            print("\n📄 Generating page images (600x400)...")
            self._generate_page_images()
            
            execution_time = (time.time() - start_time) / 60
            print(f"✅ Comic generation completed in {execution_time:.2f} minutes")
            
            return True
            
        except Exception as e:
            print(f"❌ Comic generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _enhance_all_images(self):
        """Enhance quality of all extracted frames (simple color-preserving method)"""
        if not os.path.exists(self.frames_dir):
            print(f"❌ Frames directory not found: {self.frames_dir}")
            return
        
        try:
            from backend.simple_color_enhancer import SimpleColorEnhancer
            enhancer = SimpleColorEnhancer()
            enhancer.enhance_batch(self.frames_dir)
        except Exception as e:
            print(f"❌ Simple enhancement failed: {e}")
    
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
                    
                    # The enhancement is successful if no exception was thrown
                    print(f"✅ Advanced enhancement completed: {frame_file}")
                        
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
            
            # Update to ensure we use the full story extraction count
            if hasattr(self, '_filtered_count') and self._filtered_count > 12:
                # We have full story extraction
                pass
            else:
                # Old filtering is being used, update it
                self._filtered_count = min(48, len(all_subs))
            
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
            
            # DISABLED: Don't filter in bubble generation - use all selected frames
            # filtered_subs = self._filter_meaningful_subtitles(srt_path)
            
            # Use all subtitles that were selected for frames
            with open(srt_path, 'r', encoding='utf-8') as f:
                subs = list(srt.parse(f.read()))
            
            # If we have the full story count, use only those
            if hasattr(self, '_filtered_count') and self._filtered_count > 0:
                # Take only the subtitles that correspond to our frames
                # This should match the 48 selected in story extraction
                step = len(subs) / self._filtered_count if len(subs) > self._filtered_count else 1
                filtered_subs = []
                for i in range(min(self._filtered_count, len(subs))):
                    idx = int(i * step) if step > 1 else i
                    if idx < len(subs):
                        filtered_subs.append(subs[idx])
                subs = filtered_subs
                print(f"💬 Using {len(subs)} subtitles for bubbles (matching frame count)")
            
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
        # Use 2x2 grid with 12 PAGES at 600x400 resolution
        from backend.fixed_12_pages_600x400 import generate_12_pages_600x400
        
        print(f"📖 Generating 12-page comic (600x400 resolution)")
        print(f"📊 Target: 48 meaningful panels from {len(frame_files)} frames")
        
        return generate_12_pages_600x400(frame_files, bubbles)
        
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
        """Generate smart comic with enhanced emotion matching"""
        try:
            # Use enhanced emotion matching
            from backend.enhanced_emotion_matcher import EnhancedEmotionMatcher
            from backend.eye_state_detector import EyeStateDetector
            
            if not os.path.exists('test1.srt'):
                print("❌ Missing subtitles for smart comic")
                return
            
            print("🎭 Generating Smart Comic with Enhanced Features...")
            print("  👁️ Eye detection: Avoiding half-closed eyes")
            print("  😊 Emotion matching: Text ↔ Facial expressions")
            
            # Initialize components
            emotion_matcher = EnhancedEmotionMatcher()
            eye_detector = EyeStateDetector() if emotion_match else None
            
            # Get frames and subtitles
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            frame_paths = [os.path.join(self.frames_dir, f) for f in frame_files]
            
            with open('test1.srt', 'r', encoding='utf-8') as f:
                import srt
                subtitles = list(srt.parse(f.read()))
            
            # Use filtered subtitles if available
            if hasattr(self, '_filtered_count') and self._filtered_count > 0:
                step = len(subtitles) / self._filtered_count if len(subtitles) > self._filtered_count else 1
                filtered_subtitles = []
                for i in range(min(self._filtered_count, len(subtitles))):
                    idx = int(i * step) if step > 1 else i
                    if idx < len(subtitles):
                        filtered_subtitles.append(subtitles[idx])
                subtitles = filtered_subtitles[:len(frame_paths)]  # Match frame count
            
            print(f"  📝 Analyzing {len(subtitles)} dialogues")
            
            # Match frames to emotions
            matched_panels = emotion_matcher.match_frames_to_emotions(
                frame_paths[:len(subtitles)], subtitles, eye_detector
            )
            
            print(f"  ✅ Created {len(matched_panels)} emotion-matched panels")
            
            # Generate smart comic data
            comic_data = {
                'title': 'Emotion-Aware Comic',
                'panels': []
            }
            
            for i, panel in enumerate(matched_panels):
                # Get dominant emotions
                text_emotion = max(panel['text_emotions'].items(), 
                                 key=lambda x: x[1] if x[0] != 'intensity' else 0)[0]
                face_emotion = max(panel['face_emotions'].items(), 
                                 key=lambda x: x[1] if x[0] != 'intensity' else 0)[0]
                
                comic_data['panels'].append({
                    'id': i,
                    'frame': os.path.basename(panel['frame']),
                    'text': panel['subtitle'].content,
                    'text_emotion': text_emotion,
                    'face_emotion': face_emotion,
                    'match_score': panel['match_score'],
                    'eye_score': panel.get('eye_score', 1.0),
                    'emotions': {
                        'text': panel['text_emotions'],
                        'face': panel['face_emotions']
                    }
                })
            
            # Save and generate viewer
            if comic_data['panels']:
                self._generate_smart_viewer(comic_data)
                print("✅ Smart comic generated: output/smart_comic_viewer.html")
            else:
                print("❌ No panels generated for smart comic")
                
        except Exception as e:
            print(f"⚠️ Smart comic generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_smart_viewer(self, comic_data):
        """Generate HTML viewer for smart comic"""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Smart Comic - Emotion Matched</title>
    <style>
        body { margin: 0; padding: 20px; background: #2c3e50; color: white; font-family: Arial, sans-serif; }
        .header { text-align: center; margin-bottom: 30px; }
        .comic-container { max-width: 1200px; margin: 0 auto; }
        .comic-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0; margin-top: 30px; }
        .comic-panel { background: white; border: 4px solid #333; box-shadow: 0 5px 20px rgba(0,0,0,0.3); position: relative; overflow: hidden; }
        .comic-panel img { width: 100%; height: 200px; object-fit: cover; display: block; }
        .panel-info { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.8); color: white; padding: 15px; }
        .panel-text { font-size: 14px; margin-bottom: 8px; line-height: 1.4; }
        .emotion-badges { display: flex; gap: 10px; font-size: 12px; }
        .emotion-badge { padding: 4px 8px; border-radius: 12px; font-weight: bold; }
        .emotion-happy { background: #4CAF50; color: white; }
        .emotion-sad { background: #2196F3; color: white; }
        .emotion-angry { background: #F44336; color: white; }
        .emotion-surprised { background: #FF9800; color: white; }
        .emotion-scared { background: #9C27B0; color: white; }
        .emotion-neutral { background: #666; color: white; }
        .match-score { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.7); color: white; padding: 5px 10px; border-radius: 5px; font-size: 12px; }
        .good-match { background: #4CAF50; }
        .medium-match { background: #FF9800; }
        .poor-match { background: #F44336; }
        h2 { text-align: center; margin: 30px 0 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎭 Smart Comic Summary</h1>
        <p>AI-generated comic with emotion matching and eye quality detection</p>
        <p style="font-size: 14px; color: #bbb;">''' + str(len(comic_data.get('panels', []))) + ''' key panels selected from the story</p>
    </div>
    
    <div class="comic-container">
        <div class="comic-grid">
'''
        
        # Generate panels in grid layout
        panels = comic_data.get('panels', [])
        for i, panel in enumerate(panels):
            # Determine match quality
            match_score = panel.get('match_score', 0)
            match_class = 'good-match' if match_score > 0.7 else 'medium-match' if match_score > 0.4 else 'poor-match'
            
            html += f'''
            <div class="comic-panel">
                <img src="/frames/final/{panel['frame']}" alt="Panel {i+1}" onerror="this.src='/frames/final/frame{i:03d}.png'">
                <div class="match-score {match_class}">
                    Match: {match_score:.1%} | Eyes: {panel.get('eye_score', 1.0):.1%}
                </div>
                <div class="panel-info">
                    <div class="panel-text">{panel['text']}</div>
                    <div class="emotion-badges">
                        <span class="emotion-badge emotion-{panel['text_emotion']}">Text: {panel['text_emotion']}</span>
                        <span class="emotion-badge emotion-{panel['face_emotion']}">Face: {panel['face_emotion']}</span>
                    </div>
                </div>
            </div>
'''
        
        html += '''
        </div>
        
        <h2>📊 Emotion Analysis Summary</h2>
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin: 20px 0;">
'''
        
        # Add summary statistics
        if panels:
            # Count emotion matches
            perfect_matches = sum(1 for p in panels if p['text_emotion'] == p['face_emotion'])
            good_eye_scores = sum(1 for p in panels if p.get('eye_score', 0) > 0.8)
            
            html += f'''
            <p>✅ Perfect emotion matches: {perfect_matches}/{len(panels)} ({perfect_matches/len(panels)*100:.0f}%)</p>
            <p>👁️ Panels with open eyes: {good_eye_scores}/{len(panels)} ({good_eye_scores/len(panels)*100:.0f}%)</p>
            <p>📈 Average match score: {sum(p.get('match_score', 0) for p in panels)/len(panels):.1%}</p>
'''
        
        html += '''
        </div>
    </div>
</body>
</html>'''
        
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
    
    def _generate_page_images(self):
        """Generate page images at 600x400 resolution"""
        try:
            from backend.page_image_generator import PageImageGenerator
            
            # Create generator
            generator = PageImageGenerator(output_dir="output/page_images")
            
            # Load pages data
            pages_json_path = "output/pages.json"
            if not os.path.exists(pages_json_path):
                print("⚠️ Pages JSON not found, skipping page image generation")
                return
            
            with open(pages_json_path, 'r') as f:
                pages_data = json.load(f)
            
            # Generate images
            saved_pages = generator.generate_page_images(pages_data, "frames/final")
            
            if saved_pages:
                print(f"✅ Generated {len(saved_pages)} page images (600x400)")
                print("📄 Page gallery available at: output/page_images/index.html")
                
                # Open the gallery in browser
                gallery_url = f"http://localhost:5000/output/page_images/index.html"
                print(f"🌐 View page images at: {gallery_url}")
        
        except Exception as e:
            print(f"⚠️ Page image generation failed: {e}")
    
    def _copy_template_files(self):
        """Copy template files to output directory"""
        try:
            # Copy HTML template with editing functionality
            template_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Comic - Interactive Editor</title>
    <style>
        body { margin: 0; padding: 20px; background: #f0f0f0; font-family: Arial, sans-serif; }
        .comic-container { max-width: 1200px; margin: 0 auto; }
        .comic-page { 
            background: white; 
            width: 600px; /* Exact image width */
            height: 400px; /* Exact image height */
            padding: 0; /* No padding */
            margin: 0; /* No margin */
            box-shadow: 0 0 10px rgba(0,0,0,0.1); 
            box-sizing: content-box; /* Don't include border in size */
            position: relative;
            overflow: hidden;
        }
        .comic-grid { 
            display: grid !important; 
            grid-template-columns: 295px 295px !important; 
            grid-template-rows: 195px 195px !important; 
            gap: 10px !important; /* 10px gap everywhere */
            width: 600px !important;
            height: 400px !important;
            margin: 0 !important;
            padding: 0 !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            background: white !important; /* White divider color */
            box-sizing: border-box !important;
        }
        .page-wrapper {
            margin: 30px auto;
            width: 600px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .page-title { 
            text-align: center; 
            color: #333; 
            margin-bottom: 10px; 
            font-size: 18px; 
            font-weight: bold; 
        }
        .page-info {
            position: absolute;
            bottom: 10px;
            right: 15px;
            font-size: 14px;
            color: #666;
            font-weight: bold;
            background: rgba(255, 255, 255, 0.9);
            padding: 5px 10px;
            border-radius: 3px;
            border: 1px solid #ddd;
            z-index: 10;
            font-family: monospace;
        }
        .page-info.top-left {
            bottom: auto;
            right: auto;
            top: 10px;
            left: 15px;
        }
        .page-info.top-right {
            bottom: auto;
            top: 10px;
        }
        .page-info.bottom-left {
            right: auto;
            left: 15px;
        }
        .panel { 
            position: relative !important; 
            border: none !important;
            overflow: hidden !important;
            width: 295px !important; /* Exact panel size for 10px gaps */
            height: 195px !important; /* Exact panel size for 10px gaps */
            box-sizing: border-box !important;
            margin: 0 !important;
            padding: 0 !important;
            flex-shrink: 0 !important;
            background: #f5f5f5 !important; /* Light background to see panels clearly */
        }
        /* All individual panel borders removed for perfect zero gaps */
        .panel img { 
            width: 100%; 
            height: 100%; 
            object-fit: contain; /* No zooming - shows entire image */
            object-position: center; /* Center the image */
            background-color: #fff; /* White background for letterbox areas */
        }
        
        /* Alternative modes - uncomment one to use */
        /* .panel img { object-fit: cover; } */ /* Zoom to fill (crops edges) */
        /* .panel img { object-fit: fill; } */ /* Stretch to fit (may distort) */
        /* .panel img { object-fit: scale-down; } */ /* Shrink if needed */
        
        /* Exact 600x400 mode - no individual borders */
        .exact-size .panel { 
            border: none !important; 
        }
        .comic-grid { 
            border: 2px solid #333; /* Single outer border only */
            box-sizing: border-box;
        }
        
        /* Unity export mode - no borders, clean images */
        .unity-export .panel { 
            border: none !important; 
        }
        .unity-export .comic-grid { 
            border: none !important; 
        }
        .unity-export .page-info {
            display: none !important;
        }
        
        /* Debug mode - shows exact dimensions */
        .debug-mode .comic-page {
            outline: 2px solid red;
        }
        .debug-mode .comic-page::before {
            content: "Page: 600×400";
            position: absolute;
            top: -25px;
            left: 0;
            color: red;
            font-size: 12px;
            z-index: 100;
        }
        .debug-mode .comic-grid {
            outline: 2px solid blue;
        }
        .debug-mode .panel {
            outline: 1px solid green;
        }
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
            cursor: move;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .speech-bubble:hover { 
            transform: scale(1.02); 
            box-shadow: 3px 3px 12px rgba(0,0,0,0.6); 
        }
        .speech-bubble.editing { 
            cursor: text; 
        }
        .speech-bubble textarea {
            width: 100%;
            height: 100%;
            border: none;
            background: transparent;
            font: inherit;
            text-align: center;
            resize: none;
            outline: 2px solid #4CAF50;
            padding: 5px;
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
        .edit-controls {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.85);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            font-size: 14px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .edit-controls h4 { margin: 0 0 10px 0; color: #4CAF50; }
        .edit-controls p { margin: 5px 0; opacity: 0.9; }
    </style>
</head>
<body>
    <div class="comic-container">
        <h1 class="comic-title">🎬 Generated Comic</h1>
        <div id="comic-pages">
            <div class="loading">Loading comic...</div>
        </div>
    </div>
    
    <!-- Edit Controls -->
    <div class="edit-controls">
        <h4>✏️ Interactive Editor</h4>
        <p>• <strong>Drag</strong> speech bubbles to move</p>
        <p>• <strong>Double-click</strong> to edit text</p>
        <p>• Changes auto-save locally</p>
        <button onclick="changeBubbleInteractive()" style="margin-top: 8px; padding: 6px 12px; background: #9C27B0; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; width: 100%; font-size: 11px;">
            🎨 Change Bubble
        </button>
        <div style="display: flex; gap: 5px; margin-top: 10px;">
            <button onclick="saveEditableHTML()" style="padding: 6px 10px; background: #FF9800; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; flex: 1; font-size: 11px;">
                💾 Save
            </button>
            <button onclick="exportToPDF()" style="padding: 6px 10px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; flex: 1; font-size: 11px;">
                📄 PDF
            </button>
            <button onclick="printComic()" style="padding: 6px 10px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; flex: 1; font-size: 11px;">
                🖨️ Print
            </button>
        </div>
        <div style="display: flex; gap: 5px; margin-top: 5px;">
            <button onclick="viewPageImages()" style="padding: 6px 10px; background: #9C27B0; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; flex: 1; font-size: 11px;">
                🖼️ Pages
            </button>
            <button onclick="toggleUnityMode()" style="padding: 6px 10px; background: #FF5722; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; flex: 1; font-size: 11px;">
                🎮 Unity
            </button>
        </div>
        <button onclick="checkDimensions()" style="margin-top: 5px; padding: 6px 10px; background: #607D8B; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; width: 100%; font-size: 11px;">
            📏 Check Dimensions
        </button>
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
                            // Create wrapper for title and page
                            const pageWrapper = document.createElement('div');
                            pageWrapper.className = 'page-wrapper';
                            
                            // Add page title outside the page
                            const pageTitle = document.createElement('h2');
                            pageTitle.className = 'page-title';
                            pageTitle.textContent = `Page ${pageIndex + 1}`;
                            pageWrapper.appendChild(pageTitle);
                            
                            // Create page container (exact 600x400)
                            const pageDiv = document.createElement('div');
                            pageDiv.className = 'comic-page';
                            
                            // Add page info (resolution)
                            const pageInfo = document.createElement('div');
                            pageInfo.className = 'page-info';
                            pageInfo.textContent = '600x400';
                            pageDiv.appendChild(pageInfo);
                            
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
                            
                            // Add page to wrapper, then wrapper to container
                            pageWrapper.appendChild(pageDiv);
                            pagesContainer.appendChild(pageWrapper);
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
            
        // Initialize editing functionality after comic loads
        setTimeout(initializeEditor, 1000);
        
        // Editing functionality
        let currentEditBubble = null;
        let draggedBubble = null;
        let offset = {x: 0, y: 0};
        
        function initializeEditor() {
            document.querySelectorAll('.speech-bubble').forEach(bubble => {
                bubble.addEventListener('dblclick', (e) => {
                    e.stopPropagation();
                    editBubbleText(bubble);
                });
                bubble.addEventListener('mousedown', startDrag);
            });
            
            document.addEventListener('mousemove', drag);
            document.addEventListener('mouseup', stopDrag);
            loadSavedState();
        }
        
        function editBubbleText(bubble) {
            if (currentEditBubble) return;
            
            currentEditBubble = bubble;
            bubble.classList.add('editing');
            
            const text = bubble.innerText;
            const textarea = document.createElement('textarea');
            textarea.value = text;
            
            bubble.innerHTML = '';
            bubble.appendChild(textarea);
            textarea.focus();
            textarea.select();
            
            textarea.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    saveBubbleText(bubble, textarea.value);
                }
                if (e.key === 'Escape') {
                    saveBubbleText(bubble, text);
                }
            });
            
            textarea.addEventListener('blur', () => {
                setTimeout(() => {
                    if (currentEditBubble === bubble) {
                        saveBubbleText(bubble, textarea.value);
                    }
                }, 100);
            });
        }
        
        function saveBubbleText(bubble, text) {
            bubble.innerText = text;
            bubble.classList.remove('editing');
            currentEditBubble = null;
            saveState();
        }
        
        function startDrag(e) {
            if (e.target.tagName === 'TEXTAREA') return;
            
            const bubble = e.target.closest('.speech-bubble');
            if (!bubble || currentEditBubble) return;
            
            draggedBubble = bubble;
            const rect = bubble.getBoundingClientRect();
            offset.x = e.clientX - rect.left;
            offset.y = e.clientY - rect.top;
            
            bubble.style.opacity = '0.9';
            bubble.style.zIndex = '100';
            e.preventDefault();
        }
        
        function drag(e) {
            if (!draggedBubble) return;
            
            const parent = draggedBubble.parentElement;
            const parentRect = parent.getBoundingClientRect();
            
            let x = e.clientX - parentRect.left - offset.x;
            let y = e.clientY - parentRect.top - offset.y;
            
            x = Math.max(0, Math.min(x, parentRect.width - draggedBubble.offsetWidth));
            y = Math.max(0, Math.min(y, parentRect.height - draggedBubble.offsetHeight));
            
            draggedBubble.style.left = x + 'px';
            draggedBubble.style.top = y + 'px';
        }
        
        function stopDrag() {
            if (draggedBubble) {
                draggedBubble.style.opacity = '';
                draggedBubble.style.zIndex = '';
                saveState();
                draggedBubble = null;
            }
        }
        
        function saveState() {
            const bubbles = [];
            document.querySelectorAll('.speech-bubble').forEach((bubble, index) => {
                bubbles.push({
                    index: index,
                    text: bubble.innerText,
                    left: bubble.style.left,
                    top: bubble.style.top
                });
            });
            localStorage.setItem('comicBubbles', JSON.stringify(bubbles));
        }
        
        function loadSavedState() {
            const saved = localStorage.getItem('comicBubbles');
            if (!saved) return;
            
            try {
                const bubbles = JSON.parse(saved);
                const elements = document.querySelectorAll('.speech-bubble');
                
                bubbles.forEach((data, index) => {
                    if (elements[index]) {
                        elements[index].innerText = data.text;
                        if (data.left) elements[index].style.left = data.left;
                        if (data.top) elements[index].style.top = data.top;
                    }
                });
            } catch (e) {
                console.error('Failed to load saved state:', e);
            }
        }
        
        // Export functions
        function printComic() {
            // Hide edit controls for printing
            document.querySelector('.edit-controls').style.display = 'none';
            
            // Use browser's print function
            window.print();
            
            // Show edit controls again
            setTimeout(() => {
                document.querySelector('.edit-controls').style.display = 'block';
            }, 100);
        }
        
        // View page images gallery
        function viewPageImages() {
            window.open('/output/page_images/index.html', '_blank');
        }
        
        // Toggle Unity export mode (no borders)
        let unityMode = false;
        function toggleUnityMode() {
            unityMode = !unityMode;
            const container = document.querySelector('.comic-container');
            
            if (unityMode) {
                container.classList.add('unity-export');
                showSaveMessage('🎮 Unity Mode ON - Borders hidden for clean export');
                
                // Update button text
                event.target.innerHTML = '🎮 Unity Mode ON (Click to disable)';
                event.target.style.background = '#4CAF50';
            } else {
                container.classList.remove('unity-export');
                showSaveMessage('📚 Normal Mode - Borders visible');
                
                // Update button text
                event.target.innerHTML = '🎮 Unity Mode (No Borders)';
                event.target.style.background = '#FF5722';
            }
        }
        
        // Check exact dimensions
        function checkDimensions() {
            const pages = document.querySelectorAll('.comic-page');
            const container = document.querySelector('.comic-container');
            
            // Toggle debug mode and exact-size mode
            container.classList.toggle('debug-mode');
            container.classList.toggle('exact-size');
            
            // Get first page dimensions
            if (pages.length > 0) {
                const page = pages[0];
                const grid = page.querySelector('.comic-grid');
                const pageRect = page.getBoundingClientRect();
                const gridRect = grid ? grid.getBoundingClientRect() : null;
                const computed = window.getComputedStyle(page);
                
                const info = `📏 Page Dimensions Check:\n\n` +
                    `Page Width: ${pageRect.width}px (should be 600)\n` +
                    `Page Height: ${pageRect.height}px (should be 400)\n` +
                    `Grid Width: ${gridRect ? gridRect.width : 'N/A'}px\n` +
                    `Grid Height: ${gridRect ? gridRect.height : 'N/A'}px\n` +
                    `Padding: ${computed.padding}\n` +
                    `Box-sizing: ${computed.boxSizing}\n\n` +
                    `${pageRect.width === 600 && pageRect.height === 400 ? '✅ EXACT MATCH!' : '❌ Size mismatch!'}\n\n` +
                    `Exact-size mode: ${container.classList.contains('exact-size') ? 'ON' : 'OFF'}`;
                
                alert(info);
                
                // Update button text
                const btn = event.target;
                if (container.classList.contains('exact-size')) {
                    btn.innerHTML = '📏 Exact Mode ON';
                    btn.style.background = '#4CAF50';
                } else {
                    btn.innerHTML = '📏 Check Dimensions';
                    btn.style.background = '#607D8B';
                }
            }
        }
        
        function exportToPDF() {
            // For basic PDF export, we'll use the print dialog with PDF option
            // Most browsers support "Save as PDF" in print dialog
            
            // First, add print-specific styles
            const printStyles = document.createElement('style');
            printStyles.innerHTML = `
                @media print {
                    /* Reset all margins and padding */
                    * {
                        -webkit-print-color-adjust: exact !important;
                        print-color-adjust: exact !important;
                        color-adjust: exact !important;
                    }
                    
                    body { 
                        margin: 0 !important; 
                        padding: 0 !important;
                        background: white !important;
                    }
                    
                    /* Hide non-comic elements */
                    .edit-controls, .comic-title, .save-notice { 
                        display: none !important; 
                    }
                    
                    /* Full page for comic container */
                    .comic-container {
                        margin: 0 !important;
                        padding: 0 !important;
                        max-width: none !important;
                        width: 100% !important;
                    }
                    
                    /* Each comic page exactly 600x400 */
                    .comic-page { 
                        page-break-inside: avoid !important;
                        page-break-after: always !important;
                        margin: 0 !important;
                        padding: 0 !important;
                        box-shadow: none !important;
                        background: white !important;
                        width: 600px !important;
                        height: 400px !important;
                        box-sizing: border-box !important;
                        position: relative !important;
                    }
                    
                    /* Hide wrapper elements in print */
                    .page-wrapper {
                        page-break-inside: avoid !important;
                    }
                    .page-title {
                        display: none !important;
                    }
                    
                    /* Comic grid exact 600x400 with 4 panels */
                    .comic-grid {
                        width: 600px !important;
                        height: 400px !important;
                        margin: 0 !important;
                        padding: 0 !important;
                        column-gap: 10px !important; /* 10px horizontal divider */
                        row-gap: 10px !important; /* 10px vertical divider */
                        display: grid !important;
                        grid-template-columns: 295px 295px !important;
                        grid-template-rows: 195px 195px !important;
                        background: white !important; /* White divider color */
                    }
                    
                    /* Show page info in print */
                    .page-info {
                        display: block !important;
                        position: absolute !important;
                        bottom: 5px !important;
                        right: 10px !important;
                        font-size: 10px !important;
                        color: #999 !important;
                    }
                    
                    /* Panels exact 300x200 each - no gaps */
                    .panel {
                        width: 300px !important;
                        height: 200px !important;
                        border: none !important; /* Remove borders for zero gaps */
                        overflow: hidden !important;
                        position: relative !important;
                        box-sizing: border-box !important;
                        margin: 0 !important;
                        padding: 0 !important;
                    }
                    /* Remove double borders in print */
                    .panel:nth-child(1), .panel:nth-child(3) {
                        border-right: none !important;
                    }
                    .panel:nth-child(1), .panel:nth-child(2) {
                        border-bottom: none !important;
                    }
                    
                    .panel img {
                        width: 100% !important;
                        height: 100% !important;
                        object-fit: contain !important; /* No zooming/cropping */
                        background-color: white !important;
                    }
                    
                    /* Speech bubbles maintain position */
                    .speech-bubble { 
                        -webkit-print-color-adjust: exact !important;
                        print-color-adjust: exact !important;
                        background: white !important;
                        border: 3px solid black !important;
                    }
                    
                    /* Page settings */
                    @page { 
                        size: A4 landscape;
                        margin: 10mm;
                    }
                    
                    /* Remove last page break */
                    .comic-page:last-child {
                        page-break-after: avoid !important;
                    }
                }
            `;
            document.head.appendChild(printStyles);
            
            // Show instructions with recommended settings
            alert('📄 Export to PDF - Recommended Settings\\n\\n' +
                  '1. Destination: "Save as PDF"\\n' +
                  '2. Layout: "Landscape" (for better fit)\\n' +
                  '3. Paper size: "A4" or "Letter"\\n' +
                  '4. Margins: "Default" or "None"\\n' +
                  '5. Scale: "Default (100%)" or "Fit to page"\\n' +
                  '6. Options: ✓ "Background graphics"\\n\\n' +
                  'Click Save to create your PDF!');
            
            // Trigger print
            printComic();
        }
        
        // Add keyboard shortcut for export
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
                e.preventDefault();
                exportToPDF();
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                saveEditableHTML();
            }
        });
        
        // Save editable HTML with all current edits
        function saveEditableHTML() {
            // Update the current DOM with edited content
            const currentState = {
                bubbles: [],
                timestamp: new Date().toISOString()
            };
            
            // Collect current bubble states
            document.querySelectorAll('.speech-bubble').forEach((bubble, index) => {
                currentState.bubbles.push({
                    text: bubble.innerText,
                    left: bubble.style.left,
                    top: bubble.style.top
                });
            });
            
            // Clone the current document
            const docClone = document.documentElement.cloneNode(true);
            
            // Remove the loading message from clone
            const loadingDiv = docClone.querySelector('.loading');
            if (loadingDiv) loadingDiv.remove();
            
            // Add a marker to show this is a saved version
            const savedNotice = docClone.createElement('div');
            savedNotice.style.cssText = 'position: fixed; top: 10px; left: 10px; background: #4CAF50; color: white; padding: 10px; border-radius: 5px; z-index: 1000;';
            savedNotice.innerHTML = '✅ This is a saved editable comic - Continue editing anytime!';
            docClone.body.insertBefore(savedNotice, docClone.body.firstChild);
            
            // Inject the current state into the saved file
            const stateScript = docClone.createElement('script');
            stateScript.innerHTML = `
                // Saved state from ${new Date().toLocaleString()}
                const savedState = ${JSON.stringify(currentState)};
                
                // Auto-restore saved state when file opens
                window.addEventListener('load', () => {
                    setTimeout(() => {
                        const bubbles = document.querySelectorAll('.speech-bubble');
                        savedState.bubbles.forEach((state, index) => {
                            if (bubbles[index]) {
                                bubbles[index].innerText = state.text;
                                if (state.left) bubbles[index].style.left = state.left;
                                if (state.top) bubbles[index].style.top = state.top;
                            }
                        });
                        console.log('✅ Restored saved edits from', savedState.timestamp);
                    }, 1500);
                });
            `;
            docClone.head.appendChild(stateScript);
            
            // Convert to string
            const htmlContent = '<!DOCTYPE html>\\n' + docClone.outerHTML;
            
            // Create blob and download
            const blob = new Blob([htmlContent], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            
            // Generate filename with timestamp
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
            a.download = `comic_editable_${timestamp}.html`;
            
            a.click();
            URL.revokeObjectURL(url);
            
            // Show success message
            showSaveMessage('✅ Comic saved! You can open this HTML file anytime to continue editing.');
        }
        
        // Show temporary save message
        function showSaveMessage(message) {
            const msgDiv = document.createElement('div');
            msgDiv.style.cssText = 'position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #4CAF50; color: white; padding: 20px 30px; border-radius: 10px; font-size: 16px; z-index: 10000; box-shadow: 0 4px 20px rgba(0,0,0,0.3);';
            msgDiv.innerHTML = message;
            document.body.appendChild(msgDiv);
            
            setTimeout(() => {
                msgDiv.style.transition = 'opacity 0.5s';
                msgDiv.style.opacity = '0';
                setTimeout(() => msgDiv.remove(), 500);
            }, 3000);
        }
        
        function changeBubbleInteractive() {
            // COMPLETELY NEW APPROACH: Direct DOM targeting
            
            console.log('🚀 Starting new bubble change approach...');
            
            // Step 1: Show bubble type selection
            const bubbleTypes = ['normal', 'jagged', 'thought', 'idea', 'boom', 'square', 'whisper', 'scream', 'dream', 'radio', 'electric', 'love', 'crystal', 'fire', 'empty'];
            const bubbleEmojis = ['💬', '⚡', '💭', '💡', '💥', '📝', '🤫', '😱', '🌙', '📡', '⚡', '💖', '💎', '🔥', '❌'];
            const bubbleNames = ['Normal', 'Jagged', 'Thought', 'Idea', 'Boom', 'Square', 'Whisper', 'Scream', 'Dream', 'Radio', 'Electric', 'Love', 'Crystal', 'Fire', 'Empty'];
            
            let menu = '🎨 SELECT BUBBLE TYPE:\\n\\n';
            for (let i = 0; i < bubbleTypes.length; i++) {
                menu += `${i + 1}. ${bubbleEmojis[i]} ${bubbleNames[i]}\\n`;
            }
            
            const choice = prompt(menu + '\\nEnter 1-15:');
            const choiceNum = parseInt(choice);
            
            if (isNaN(choiceNum) || choiceNum < 1 || choiceNum > 15) {
                alert('❌ Invalid choice! Please enter 1-15.');
                return;
            }
            
            const selectedType = bubbleTypes[choiceNum - 1];
            const selectedEmoji = bubbleEmojis[choiceNum - 1];
            const selectedName = bubbleNames[choiceNum - 1];
            
            // Step 2: Ask for panel number (1-48)
            const panelPrompt = '🎯 ENTER PANEL NUMBER (1-48):\\n\\n' +
                '📖 Comic has 12 pages × 4 panels = 48 total\\n\\n' +
                'Page 1: Panels 1, 2, 3, 4\\n' +
                'Page 2: Panels 5, 6, 7, 8\\n' +
                'Page 3: Panels 9, 10, 11, 12\\n' +
                '...and so on...\\n' +
                'Page 12: Panels 45, 46, 47, 48';
            
            const panelChoice = prompt(panelPrompt);
            const panelNum = parseInt(panelChoice);
            
            if (isNaN(panelNum) || panelNum < 1 || panelNum > 48) {
                alert('❌ Invalid panel! Enter 1-48.');
                return;
            }
            
            // Step 3: Apply using completely new method (no size selection needed)
            const result = directBubbleChange(panelNum, selectedType);
            
            if (result.success) {
                alert(`✅ SUCCESS!\\n\\nPanel ${panelNum} bubble changed to ${selectedEmoji} ${selectedName}\\n\\nLocation: ${result.location}\\n\\n💡 RESIZE: Hold and drag the bubble corners to stretch manually!`);
                
                // Add resize instructions
                setTimeout(() => {
                    alert('🎯 MANUAL RESIZE INSTRUCTIONS:\\n\\n1. Hover over the bubble\\n2. Hold and drag any corner to stretch\\n3. Resize in real-time by dragging\\n4. Release to set new size\\n\\n✨ Try it now!');
                }, 2000);
            } else {
                alert(`❌ FAILED!\\n\\nCould not change Panel ${panelNum}\\n\\nReason: ${result.reason}`);
            }
        }
        
        function applyBubbleChange(panelNumber, bubbleType) {
            try {
                // Find all comic pages
                const pages = document.querySelectorAll('.comic-page');
                
                // Also check for grid items if comic pages not found
                const gridItems = document.querySelectorAll('.grid-item');
                
                let changed = false;
                
                // Method 1: Try comic pages approach
                pages.forEach(page => {
                    const panels = page.querySelectorAll('.panel');
                    if (panels[panelNumber - 1]) {
                        const targetPanel = panels[panelNumber - 1];
                        const bubble = targetPanel.querySelector('.speech-bubble') || targetPanel.querySelector('.bubble');
                        
                        if (bubble) {
                            // Remove all existing shape classes
                            const shapes = ['normal', 'jagged', 'thought', 'idea', 'boom', 'square'];
                            shapes.forEach(shape => bubble.classList.remove(shape));
                            
                            // Handle empty bubble type
                            if (bubbleType === 'empty') {
                                bubble.style.display = 'none';
                            } else {
                                bubble.style.display = 'flex';
                                bubble.classList.add(bubbleType);
                            }
                            
                            changed = true;
                            console.log(`✅ Changed panel ${panelNumber} bubble to ${bubbleType} (pages method)`);
                        }
                    }
                });
                
                // Method 2: Try grid items approach if no pages found
                if (!changed && gridItems.length >= panelNumber) {
                    const targetGridItem = gridItems[panelNumber - 1];
                    const bubble = targetGridItem.querySelector('.bubble') || targetGridItem.querySelector('.speech-bubble');
                    
                    if (bubble) {
                        // Remove all existing shape classes
                        const shapes = ['normal', 'jagged', 'thought', 'idea', 'boom', 'square'];
                        shapes.forEach(shape => bubble.classList.remove(shape));
                        
                        // Handle empty bubble type
                        if (bubbleType === 'empty') {
                            bubble.style.display = 'none';
                        } else {
                            bubble.style.display = 'flex';
                            bubble.classList.add(bubbleType);
                        }
                        
                        changed = true;
                        console.log(`✅ Changed panel ${panelNumber} bubble to ${bubbleType} (grid method)`);
                    }
                }
                
                // Method 3: Debug - log what elements were found
                if (!changed) {
                    console.log('🔍 Debug info:');
                    console.log('Comic pages found:', pages.length);
                    console.log('Grid items found:', gridItems.length);
                    console.log('Looking for panel:', panelNumber);
                    
                    if (gridItems.length >= panelNumber) {
                        const targetItem = gridItems[panelNumber - 1];
                        console.log('Target grid item:', targetItem);
                        console.log('Bubbles in target:', targetItem.querySelectorAll('.bubble, .speech-bubble'));
                    }
                }
                
                return changed;
            } catch (error) {
                console.error('Error changing bubble:', error);
                return false;
            }
        }
        
        function directBubbleChange(globalPanelNumber, bubbleType) {
            // ULTRA-SIMPLE DIRECT APPROACH
            console.log(`🚀 DIRECT: Changing panel ${globalPanelNumber} to ${bubbleType}`);
            
            try {
                // Method 1: Find all speech bubbles directly
                const allSpeechBubbles = document.querySelectorAll('.speech-bubble');
                console.log(`Found ${allSpeechBubbles.length} speech bubbles total`);
                
                if (allSpeechBubbles.length >= globalPanelNumber) {
                    const targetBubble = allSpeechBubbles[globalPanelNumber - 1];
                    console.log(`Targeting speech bubble #${globalPanelNumber}:`, targetBubble);
                    
                    // Direct style application
                    const shapes = ['normal', 'jagged', 'thought', 'idea', 'boom', 'square'];
                    shapes.forEach(s => targetBubble.classList.remove(s));
                    
                    if (bubbleType === 'empty') {
                        targetBubble.style.display = 'none';
                    } else {
                        targetBubble.style.display = 'flex';
                        
                        // Remove all shape classes first
                        const allShapes = ['normal', 'jagged', 'thought', 'idea', 'boom', 'square', 'whisper', 'scream', 'dream', 'radio', 'electric', 'love', 'crystal', 'fire'];
                        allShapes.forEach(shape => targetBubble.classList.remove(shape));
                        
                        // Add new bubble type class
                        targetBubble.classList.add(bubbleType);
                        
                        // Enable manual resizing
                        targetBubble.style.resize = 'both';
                        targetBubble.style.overflow = 'auto';
                        targetBubble.style.minWidth = '80px';
                        targetBubble.style.minHeight = '40px';
                        targetBubble.style.maxWidth = '280px';
                        targetBubble.style.maxHeight = '150px';
                        targetBubble.style.cursor = 'grab';
                        
                        // Add visual feedback for resizing
                        targetBubble.style.border = '3px solid #4CAF50';
                        targetBubble.style.boxShadow = '0 0 10px rgba(76, 175, 80, 0.3)';
                        
                        // Reset border after 2 seconds
                        setTimeout(() => {
                            if (targetBubble.classList.contains(bubbleType)) {
                                // Let CSS handle the styling
                                targetBubble.style.border = '';
                                targetBubble.style.boxShadow = '';
                            }
                        }, 2000);
                    }
                    
                    return { success: true, location: `Speech bubble #${globalPanelNumber}` };
                }
                
                // Method 2: Find all panels and look inside them
                const allPanels = document.querySelectorAll('.panel');
                console.log(`Found ${allPanels.length} panels total`);
                
                if (allPanels.length >= globalPanelNumber) {
                    const targetPanel = allPanels[globalPanelNumber - 1];
                    let bubble = targetPanel.querySelector('.speech-bubble');
                    
                    if (!bubble) {
                        // Create a new bubble
                        bubble = document.createElement('div');
                        bubble.className = 'speech-bubble';
                        bubble.style.cssText = 'position: absolute; top: 20px; right: 20px; background: white; border: 2px solid #333; border-radius: 15px; padding: 8px; font-size: 12px; display: flex; align-items: center; justify-content: center; z-index: 10; min-width: 80px; min-height: 30px;';
                        bubble.textContent = 'New!';
                        targetPanel.appendChild(bubble);
                        console.log('Created new bubble in panel');
                    }
                    
                    // Apply change
                    const shapes = ['normal', 'jagged', 'thought', 'idea', 'boom', 'square'];
                    shapes.forEach(s => bubble.classList.remove(s));
                    
                    if (bubbleType === 'empty') {
                        bubble.style.display = 'none';
                    } else {
                        bubble.style.display = 'flex';
                        bubble.classList.add(bubbleType);
                    }
                    
                    return { success: true, location: `Panel #${globalPanelNumber}` };
                }
                
                return { success: false, reason: `No panel or bubble found for #${globalPanelNumber}` };
                
            } catch (error) {
                console.error('Direct bubble change error:', error);
                return { success: false, reason: error.message };
            }
        }
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
            
            # Clean up previous files and SRT cache
            print("🧹 Cleaning previous video and subtitle files...")
            
            # Remove old video
            if os.path.exists('video/uploaded.mp4'):
                os.remove('video/uploaded.mp4')
                print("🗑️ Removed old video file")
            
            # Clean all SRT files to prevent caching issues
            from backend.srt_cleanup import clean_comic_workspace
            cleanup_result = clean_comic_workspace(".")
            print(f"🗑️ Cleaned {cleanup_result['total_deleted']} cached files")
            
            # Clean frames directory for fresh generation
            frames_dir = 'frames/final'
            if os.path.exists(frames_dir):
                import shutil
                shutil.rmtree(frames_dir)
                os.makedirs(frames_dir, exist_ok=True)
                print("🗑️ Cleared previous frames for fresh generation")
            
            # Save uploaded file
            f.save("video/uploaded.mp4")
            print(f"✅ New video saved: {f.filename}")
            print("🆕 Ready for fresh comic generation with new video")
            
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
            
            # Clean up previous files and SRT cache
            print("🧹 Cleaning previous video and subtitle files...")
            
            if os.path.exists('video/uploaded.mp4'):
                os.remove('video/uploaded.mp4')
                print("🗑️ Removed old video file")
            
            # Clean all SRT files to prevent caching issues
            from backend.srt_cleanup import clean_comic_workspace
            cleanup_result = clean_comic_workspace(".")
            print(f"🗑️ Cleaned {cleanup_result['total_deleted']} cached files")
            
            # Clean frames directory for fresh generation
            frames_dir = 'frames/final'
            if os.path.exists(frames_dir):
                import shutil
                shutil.rmtree(frames_dir)
                os.makedirs(frames_dir, exist_ok=True)
                print("🗑️ Cleared previous frames for fresh generation")
            
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
    # Handle nested paths for page_images
    if filename.startswith('page_images/'):
        return send_from_directory('output', filename)
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

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    """Generate PDF from edited comic data"""
    try:
        from backend.pdf_generator import generate_edited_pdf
        
        # Get edited data from request
        edited_data = request.get_json()
        
        # Generate PDF
        pdf_path = generate_edited_pdf(edited_data)
        
        # Send PDF file
        return send_file(pdf_path, as_attachment=True, download_name='comic_edited.pdf', mimetype='application/pdf')
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/create-portable')
def create_portable():
    """Create a self-contained HTML file with embedded images"""
    try:
        from backend.html_packager import create_portable_comic
        
        # Create portable version
        portable_path = create_portable_comic()
        
        # Send file
        return send_file(portable_path, as_attachment=True, download_name='comic_portable.html', mimetype='text/html')
        
    except Exception as e:
        print(f"Portable creation error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Comic Generator...")
    print("✨ Features:")
    print("   - AI-enhanced image processing")
    print("   - Advanced face detection")
    print("   - Smart bubble placement")
    print("   - High-quality comic styling")
    print("   - 11 professional bubble types")
    print("   - Perfect 10px panel gaps")
    print("   - Optimized 2x2 layout")
    print("")
    print("🌐 Web interface available at: http://localhost:5000")
    print("📁 Upload videos or paste YouTube links to generate comics!")
    
    # Get WSL IP for Windows Chrome access
    try:
        import socket
        hostname = socket.gethostname()
        wsl_ip = socket.gethostbyname(hostname)
        print(f"🖥️ For Windows Chrome: http://{wsl_ip}:5000")
    except:
        print("🖥️ For Windows Chrome: Use WSL IP address")
    
    print("")
    print("💡 WSL Chrome Access Fix:")
    print("   1. In WSL: ip addr show eth0 | grep inet")
    print("   2. Use that IP in Windows Chrome")
    print("   3. Or try: http://127.0.0.1:5000")
    print("")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)