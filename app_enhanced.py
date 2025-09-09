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
            
            # 2. Extract FULL story (don't skip important parts) - COMMENTED OUT FOR SPEED
            print("📖 Skipping story extraction for speed...")
            filtered_subs = None
            # COMMENTED OUT - EMOTION PROCESSING TAKES TOO LONG
            # if os.path.exists('test1.srt'):
            #     try:
            #         from backend.full_story_extractor import FullStoryExtractor
            #         extractor = FullStoryExtractor()
            #         
            #         # Get all subtitles first
            #         with open('test1.srt', 'r', encoding='utf-8') as f:
            #             all_subs = list(srt.parse(f.read()))
            #         
            #         # Convert to dict format
            #         sub_list = []
            #         for sub in all_subs:
            #             sub_list.append({
            #                 'index': sub.index,
            #                 'text': sub.content,
            #                 'start': sub.start.total_seconds(),
            #                 'end': sub.end.total_seconds()
            #             })
            #         
            #         # Save temp file
            #         os.makedirs('temp', exist_ok=True)
            #         with open('temp/all_subs.json', 'w') as f:
            #             json.dump(sub_list, f)
            #         
            #         # Extract full story (up to 48 panels)
            #         story_subs = extractor.extract_full_story('temp/all_subs.json')
            #         
            #         # Convert back to srt format
            #         filtered_subs = []
            #         for s in story_subs:
            #             # Find matching subtitle
            #             for sub in all_subs:
            #                 if sub.index == s.get('index', -1):
            #                     filtered_subs.append(sub)
            #                     break
            #         
            #         print(f"📚 Full story: {len(filtered_subs)} key moments from {len(all_subs)} total")
            #         
            #         # Store the count for later use
            #         self._filtered_count = len(filtered_subs)
            #         
            #     except Exception as e:
            #         print(f"⚠️ Full story extraction failed: {e}")
            #         filtered_subs = None
            
            # 3. Generate keyframes - SIMPLIFIED FOR SPEED
            print("🎯 Generating keyframes (simple method for speed)...")
            # COMMENTED OUT - COMPLEX FRAME SELECTION TAKES TOO LONG
            # if filtered_subs and smart_mode:
            #     # Use ENGAGING frame selection when smart mode is enabled
            #     print("✨ Selecting most engaging frames...")
            #     from backend.keyframes.keyframes_engaging import generate_keyframes_engaging
            #     success = generate_keyframes_engaging(self.video_path, filtered_subs, max_frames=48)
            #     if not success:
            #         print("⚠️ Engaging selection failed, trying smart method...")
            #         from backend.keyframes.keyframes_smart import generate_keyframes_smart
            #         success = generate_keyframes_smart(self.video_path, filtered_subs, max_frames=48)
            #         if not success:
            #             print("⚠️ Smart extraction failed, trying fixed method...")
            #             from backend.keyframes.keyframes_fixed import generate_keyframes_fixed
            #             generate_keyframes_fixed(self.video_path, filtered_subs, max_frames=48)
            # elif filtered_subs:
            #     # Use regular smart extraction (checks eyes but not emotions)
            #     from backend.keyframes.keyframes_smart import generate_keyframes_smart
            #     success = generate_keyframes_smart(self.video_path, filtered_subs, max_frames=48)
            #     if not success:
            #         from backend.keyframes.keyframes_fixed import generate_keyframes_fixed
            #         generate_keyframes_fixed(self.video_path, filtered_subs, max_frames=48)
            # else:
            #     # Fallback to simple method
            #     generate_keyframes_simple(self.video_path)
            
            # Use simple method for speed
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
            
            # 7. Generate simple layout - COMMENTED OUT FOR SPEED
            print("📐 Skipping complex layout generation...")
            layout_data = None  # Simple layout
            
            # 8. Create simple speech bubbles - COMMENTED OUT FOR SPEED
            print("💬 Creating simple speech bubbles...")
            bubbles = self._create_simple_bubbles()
            
            # 9. Generate final pages
            print("📄 Generating final pages...")
            pages = self._generate_pages(layout_data, bubbles)
            
            # 10. Save results
            print("💾 Saving results...")
            self._save_results(pages)
            
            # 11. Smart mode already applied during frame selection
            if smart_mode:
                print("✅ Smart frame selection completed")
            
            # 12. Extract individual panels - COMMENTED OUT FOR SPEED
            print("\n📸 Skipping panel extraction for speed...")
            # self._extract_panels()
            
            # 13. Generate page images - COMMENTED OUT FOR SPEED
            print("\n📄 Skipping page image generation for speed...")
            # self._generate_page_images()
            
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
    
    def _create_simple_bubbles(self):
        """Create simple speech bubbles without AI processing"""
        bubbles = []
        
        try:
            # Read subtitles if available
            if os.path.exists('test1.srt'):
                with open('test1.srt', 'r', encoding='utf-8') as f:
                    subs = list(srt.parse(f.read()))
                
                # Take first 4 subtitles
                for i in range(4):
                    if i < len(subs):
                        sub = subs[i]
                        bubble_obj = bubble(
                            bubble_offset_x=20,
                            bubble_offset_y=20,
                            lip_x=-1,
                            lip_y=-1,
                            dialog=sub.content,
                            emotion='normal'
                        )
                        bubbles.append(bubble_obj)
                    else:
                        # Create fallback bubble
                        bubble_obj = bubble(
                            bubble_offset_x=20,
                            bubble_offset_y=20,
                            lip_x=-1,
                            lip_y=-1,
                            dialog=f"Panel {i+1}",
                            emotion='normal'
                        )
                        bubbles.append(bubble_obj)
            else:
                # Create simple fallback bubbles
                for i in range(4):
                    bubble_obj = bubble(
                        bubble_offset_x=20,
                        bubble_offset_y=20,
                        lip_x=-1,
                        lip_y=-1,
                        dialog=f"Panel {i+1}",
                        emotion='normal'
                    )
                    bubbles.append(bubble_obj)
                    
        except Exception as e:
            print(f"Simple bubble creation failed: {e}")
            # Create fallback bubbles
            for i in range(4):
                bubble_obj = bubble(
                    bubble_offset_x=20,
                    bubble_offset_y=20,
                    lip_x=-1,
                    lip_y=-1,
                    dialog=f"Panel {i+1}",
                    emotion='normal'
                )
                bubbles.append(bubble_obj)
        
        return bubbles
    
    def _generate_pages(self, layout_data, bubbles):
        """Generate simple 4-panel pages"""
        pages = []
        
        try:
            frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
            print(f"📄 Creating simple 4-panel layout with {len(frame_files)} frames")
            
            # Create one page with 4 panels
            panels = []
            page_bubbles = []
            
            # Take first 4 frames (or cycle if less than 4)
            for i in range(4):
                frame_file = frame_files[i % len(frame_files)] if frame_files else 'blank.png'
                
                panel_obj = panel(
                    image=frame_file,
                    row_span=1,  # Simple 1x1 grid
                    col_span=1
                )
                panels.append(panel_obj)
                
                # Add simple bubble
                if i < len(bubbles):
                    page_bubbles.append(bubbles[i])
                else:
                    # Create simple fallback bubble
                    fallback_bubble = bubble(
                        bubble_offset_x=20,
                        bubble_offset_y=20,
                        lip_x=-1,
                        lip_y=-1,
                        dialog=f"Panel {i+1}",
                        emotion='normal'
                    )
                    page_bubbles.append(fallback_bubble)
            
            # Create single page
            page = Page(panels=panels, bubbles=page_bubbles)
            pages.append(page)
            
            print(f"✅ Created 1 page with 4 panels")
                
        except Exception as e:
            print(f"Page generation failed: {e}")
        
        return pages
    
    def _generate_story_pages(self, frame_files, bubbles):
        """Generate pages based on story extraction"""
        # Use 2x2 grid with 12 PAGES at 800x1080 resolution
        from backend.fixed_12_pages_800x1080 import generate_12_pages_800x1080
        
        print(f"📖 Generating 12-page comic (800x1080 resolution)")
        print(f"📊 Target: 48 meaningful panels from {len(frame_files)} frames")
        
        return generate_12_pages_800x1080(frame_files, bubbles)
        
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
        .comic-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 30px; margin-top: 30px; }
        .comic-panel { background: white; border: 4px solid #333; box-shadow: 0 5px 20px rgba(0,0,0,0.3); position: relative; overflow: hidden; }
        .comic-panel img { width: 100%; height: 400px; object-fit: cover; display: block; }
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
        """Generate page images at 800x1080 resolution"""
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
                print(f"✅ Generated {len(saved_pages)} page images (800x1080)")
                print("📄 Page gallery available at: output/page_images/index.html")
                
                # Open the gallery in browser
                gallery_url = f"http://localhost:5000/output/page_images/index.html"
                print(f"🌐 View page images at: {gallery_url}")
        
        except Exception as e:
            print(f"⚠️ Page image generation failed: {e}")
    
    def _copy_template_files(self):
        """Copy template files to output directory"""
        try:
            # SIMPLE HTML TEMPLATE - 4 EQUAL PARTS, 100% IMAGE COVERAGE
            template_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Comic - 4 Panel Layout</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            background: #f0f0f0; 
            font-family: Arial, sans-serif; 
            padding: 20px;
        }
        
        .comic-container { 
            max-width: 900px; 
            margin: 0 auto; 
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .comic-title { 
            text-align: center; 
            padding: 20px; 
            background: #333; 
            color: white; 
            font-size: 24px;
            font-weight: bold;
        }
        
        .comic-page { 
            width: 800px;
            height: 800px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 0;
            border: 3px solid #333;
        }
        
        .panel { 
            position: relative;
            overflow: hidden;
            background: #fff;
        }
        
        .panel img { 
            width: 100%; 
            height: 100%; 
            object-fit: cover;
            display: block;
        }
        
        .speech-bubble { 
            position: absolute; 
            background: white; 
            border: 2px solid #333; 
            border-radius: 10px; 
            padding: 8px 12px; 
            max-width: 150px; 
            font-size: 12px; 
            font-weight: bold;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
            z-index: 10;
            text-align: center;
            color: #333;
            cursor: move;
        }
        
        .speech-bubble:hover { 
            transform: scale(1.05); 
            box-shadow: 2px 2px 10px rgba(0,0,0,0.4); 
        }
        
        .speech-bubble::after { 
            content: ''; 
            position: absolute; 
            bottom: -8px; 
            left: 15px; 
            width: 0; 
            height: 0; 
            border-left: 8px solid transparent; 
            border-right: 8px solid transparent; 
            border-top: 8px solid #333; 
        }
        
        .loading { 
            text-align: center; 
            padding: 50px; 
            color: #666; 
            font-size: 18px;
        }
        
        .controls {
            text-align: center;
            padding: 20px;
            background: #f8f8f8;
        }
        
        .btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        
        .btn:hover {
            background: #45a049;
        }
    </style>
</head>
<body>
    <div class="comic-container">
        <h1 class="comic-title">🎬 Simple Comic Generator</h1>
        
        <div id="comic-content">
            <div class="loading">Loading comic...</div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="printComic()">🖨️ Print</button>
            <button class="btn" onclick="saveComic()">💾 Save</button>
            <button class="btn" onclick="reloadComic()">🔄 Reload</button>
        </div>
    </div>
    
    <script>
        // Load comic data
        fetch('/output/pages.json')
            .then(response => response.json())
            .then(data => {
                const container = document.getElementById('comic-content');
                container.innerHTML = '';
                
                if (data && data.length > 0) {
                    // Take first page only for simplicity
                    const pageData = data[0];
                    
                    if (pageData.panels && pageData.panels.length > 0) {
                        // Create simple 2x2 grid
                        const pageDiv = document.createElement('div');
                        pageDiv.className = 'comic-page';
                        
                        // Add 4 panels
                        for (let i = 0; i < 4; i++) {
                            const panelDiv = document.createElement('div');
                            panelDiv.className = 'panel';
                            
                            const img = document.createElement('img');
                            // Use available frames, cycle if needed
                            const frameIndex = i % pageData.panels.length;
                            img.src = '/frames/final/' + pageData.panels[frameIndex].image;
                            img.alt = `Panel ${i + 1}`;
                            
                            // Handle missing images
                            img.onerror = function() {
                                this.style.display = 'none';
                                panelDiv.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; background: #ddd; color: #666; font-size: 18px;">No Image</div>';
                            };
                            
                            panelDiv.appendChild(img);
                            
                            // Add speech bubble if available
                            if (pageData.bubbles && pageData.bubbles[frameIndex]) {
                                const bubble = pageData.bubbles[frameIndex];
                                const bubbleDiv = document.createElement('div');
                                bubbleDiv.className = 'speech-bubble';
                                bubbleDiv.textContent = bubble.dialog || 'Action!';
                                
                                // Position bubble
                                bubbleDiv.style.left = '20px';
                                bubbleDiv.style.top = '20px';
                                
                                panelDiv.appendChild(bubbleDiv);
                            }
                            
                            pageDiv.appendChild(panelDiv);
                        }
                        
                        container.appendChild(pageDiv);
                    } else {
                        container.innerHTML = '<div class="loading">No panels found</div>';
                    }
                } else {
                    container.innerHTML = '<div class="loading">No comic data found</div>';
                }
            })
            .catch(error => {
                console.error('Error loading comic:', error);
                document.getElementById('comic-content').innerHTML = '<div class="loading">Error loading comic: ' + error.message + '</div>';
            });
        
        // Simple functions
        function printComic() {
            window.print();
        }
        
        function saveComic() {
            const html = document.documentElement.outerHTML;
            const blob = new Blob([html], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'simple_comic.html';
            a.click();
            URL.revokeObjectURL(url);
        }
        
        function reloadComic() {
            location.reload();
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
    print("   - Optimized 2x2 layout")
    print("")
    print("🌐 Web interface available at: http://localhost:5000")
    print("📁 Upload videos or paste YouTube links to generate comics!")
    print("")
    app.run(debug=True, host='0.0.0.0', port=5000)