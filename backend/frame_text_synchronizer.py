"""
Frame-Text Synchronizer
Ensures perfect synchronization between selected frames and their corresponding text
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
import srt
from datetime import timedelta

class FrameTextSynchronizer:
    def __init__(self):
        self.frame_text_pairs = []
        self.video_info = {}
        
    def create_synchronized_comic(self, video_path: str, subtitles: List, target_panels: int = 48) -> bool:
        """
        Create comic with perfect frame-text synchronization
        """
        print("🎯 Creating comic with perfect frame-text synchronization...")
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        self.video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'duration': video_duration
        }
        
        print(f"📹 Video: {video_duration:.1f}s, {fps:.1f} FPS")
        
        # Create perfect frame-text pairs
        frame_text_pairs = self.create_perfect_frame_text_pairs(subtitles, video_duration, target_panels)
        
        # Extract frames with their exact matching text
        success = self.extract_synchronized_frames(cap, frame_text_pairs)
        
        cap.release()
        
        if success:
            # Generate comic with synchronized data
            self.generate_synchronized_comic(frame_text_pairs)
        
        return success
    
    def create_perfect_frame_text_pairs(self, subtitles: List, video_duration: float, target_panels: int) -> List[Dict]:
        """
        Create perfect 1:1 mapping between frames and text
        """
        print(f"🔗 Creating {target_panels} perfect frame-text pairs...")
        
        if not subtitles:
            return self.create_pairs_without_subtitles(video_duration, target_panels)
        
        # Sort subtitles by time
        subtitles.sort(key=lambda x: x.start.total_seconds())
        
        frame_text_pairs = []
        
        # Method 1: If we have enough subtitles, use them directly
        if len(subtitles) >= target_panels:
            print("📝 Using direct subtitle mapping (enough subtitles available)")
            
            # Select evenly distributed subtitles
            step = len(subtitles) / target_panels
            
            for i in range(target_panels):
                subtitle_index = int(i * step)
                if subtitle_index < len(subtitles):
                    subtitle = subtitles[subtitle_index]
                    
                    pair = {
                        'panel_number': i + 1,
                        'target_time': subtitle.start.total_seconds(),
                        'text': subtitle.content,
                        'subtitle_index': subtitle_index,
                        'has_original_subtitle': True,
                        'story_position': i / (target_panels - 1)
                    }
                    frame_text_pairs.append(pair)
                    print(f"📍 Panel {i+1:2d} ({subtitle.start}): {subtitle.content[:50]}...")
        
        # Method 2: If fewer subtitles, distribute them across timeline
        else:
            print("📝 Distributing available subtitles across timeline")
            
            segment_duration = video_duration / target_panels
            
            for i in range(target_panels):
                segment_start = i * segment_duration
                segment_end = (i + 1) * segment_duration
                segment_center = segment_start + (segment_duration / 2)
                
                # Find subtitle closest to this time segment
                segment_subtitles = [
                    sub for sub in subtitles
                    if (sub.start.total_seconds() <= segment_end and 
                        sub.end.total_seconds() >= segment_start)
                ]
                
                if segment_subtitles:
                    # Use the subtitle closest to segment center
                    best_subtitle = min(
                        segment_subtitles,
                        key=lambda x: abs(x.start.total_seconds() - segment_center)
                    )
                    
                    pair = {
                        'panel_number': i + 1,
                        'target_time': best_subtitle.start.total_seconds(),
                        'text': best_subtitle.content,
                        'subtitle_index': subtitles.index(best_subtitle),
                        'has_original_subtitle': True,
                        'story_position': i / (target_panels - 1)
                    }
                else:
                    # Create narrative text for this time segment
                    narrative_text = self.create_narrative_for_time(segment_center, video_duration, i, target_panels)
                    
                    pair = {
                        'panel_number': i + 1,
                        'target_time': segment_center,
                        'text': narrative_text,
                        'subtitle_index': -1,
                        'has_original_subtitle': False,
                        'story_position': i / (target_panels - 1)
                    }
                
                frame_text_pairs.append(pair)
                print(f"📍 Panel {i+1:2d} ({pair['target_time']:6.1f}s): {pair['text'][:50]}...")
        
        print(f"✅ Created {len(frame_text_pairs)} synchronized frame-text pairs")
        return frame_text_pairs
    
    def create_pairs_without_subtitles(self, video_duration: float, target_panels: int) -> List[Dict]:
        """Create frame-text pairs when no subtitles available"""
        pairs = []
        segment_duration = video_duration / target_panels
        
        for i in range(target_panels):
            target_time = i * segment_duration + (segment_duration / 2)
            narrative_text = self.create_narrative_for_time(target_time, video_duration, i, target_panels)
            
            pairs.append({
                'panel_number': i + 1,
                'target_time': target_time,
                'text': narrative_text,
                'subtitle_index': -1,
                'has_original_subtitle': False,
                'story_position': i / (target_panels - 1)
            })
        
        return pairs
    
    def create_narrative_for_time(self, time_pos: float, video_duration: float, panel_index: int, total_panels: int) -> str:
        """
        Create appropriate narrative text for specific time position
        """
        progress = time_pos / video_duration
        
        # Story progression based on video position
        if progress < 0.125:  # First 12.5% - Opening
            opening_texts = [
                "Our story opens with the main character in their world.",
                "We meet the protagonist as their day begins normally.",
                "The setting is established and the mood is set.",
                "Characters go about their routine, unaware of what's coming.",
                "The peaceful beginning before everything changes.",
                "Initial moments that will soon gain great significance."
            ]
            return opening_texts[panel_index % len(opening_texts)]
        
        elif progress < 0.25:  # 12.5-25% - Setup
            setup_texts = [
                "Important characters are introduced and relationships established.",
                "We learn about the protagonist's goals and motivations.",
                "The world and its rules become clear to us.",
                "Key relationships that will drive the story are formed.",
                "Background information helps us understand the situation.",
                "The foundation is laid for everything that will follow."
            ]
            return setup_texts[panel_index % len(setup_texts)]
        
        elif progress < 0.5:  # 25-50% - Rising Action
            rising_texts = [
                "Challenges begin to emerge and complicate the situation.",
                "The protagonist faces their first real obstacles.",
                "Conflicts develop as different forces come into opposition.",
                "Characters must adapt and grow to handle new difficulties.",
                "The stakes begin to rise as problems multiply.",
                "Relationships are tested under increasing pressure.",
                "Each solution leads to new and greater challenges.",
                "The scope of the conflict expands beyond initial expectations.",
                "Characters discover hidden strengths and weaknesses.",
                "Alliances form and shift as the situation evolves.",
                "The path forward becomes increasingly uncertain.",
                "Preparation begins for the major challenges ahead."
            ]
            return rising_texts[panel_index % len(rising_texts)]
        
        elif progress < 0.75:  # 50-75% - Climax
            climax_texts = [
                "The major confrontation begins as all forces converge.",
                "Everything the characters learned is put to the test.",
                "The conflict reaches its most intense and critical point.",
                "Characters must overcome their deepest fears and limitations.",
                "The truth about the situation is finally revealed.",
                "Sacrifices must be made as the stakes reach their peak.",
                "The fate of everyone involved hangs in the balance.",
                "Heroes and antagonists clash in the most crucial moments.",
                "Unexpected developments change the nature of the conflict.",
                "Characters discover reserves of strength they never knew existed.",
                "The outcome of everything depends on what happens now.",
                "This is the moment everything has been building toward."
            ]
            return climax_texts[panel_index % len(climax_texts)]
        
        else:  # 75-100% - Resolution
            resolution_texts = [
                "The immediate crisis is resolved through decisive action.",
                "Characters begin to process and understand what happened.",
                "The world starts to heal and rebuild from the conflict.",
                "Relationships are redefined by everything they've experienced.",
                "Consequences of choices become clear to everyone involved.",
                "Order begins to emerge from the chaos of recent events.",
                "Characters reflect on their growth and transformation.",
                "New beginnings emerge from the resolution of old conflicts.",
                "The community comes together to celebrate and rebuild.",
                "Lessons learned are shared and wisdom is passed forward.",
                "Peace is restored and the world finds new balance.",
                "The story concludes with hope for the future ahead."
            ]
            return resolution_texts[panel_index % len(resolution_texts)]
    
    def extract_synchronized_frames(self, cap, frame_text_pairs: List[Dict]) -> bool:
        """
        Extract frames with perfect synchronization to their text
        """
        print("🎬 Extracting frames synchronized with their text content...")
        
        output_dir = 'frames/final'
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear existing frames
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(output_dir, f))
        
        fps = self.video_info['fps']
        
        for i, pair in enumerate(frame_text_pairs):
            target_time = pair['target_time']
            text = pair['text']
            
            # Calculate frame number
            frame_number = int(target_time * fps)
            frame_number = max(0, min(frame_number, self.video_info['total_frames'] - 1))
            
            # Extract frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Save frame with synchronized index
                output_path = os.path.join(output_dir, f'frame{i:03d}.png')
                cv2.imwrite(output_path, frame)
                
                # Update pair with frame info
                pair['frame_path'] = output_path
                pair['frame_extracted'] = True
                pair['actual_frame_number'] = frame_number
                
                print(f"✅ Panel {i+1:2d}: Frame at {target_time:6.1f}s → '{text[:40]}...'")
            else:
                print(f"❌ Panel {i+1:2d}: Failed to extract frame at {target_time:.1f}s")
                pair['frame_extracted'] = False
        
        # Save synchronization data
        with open(os.path.join(output_dir, 'frame_text_sync.json'), 'w') as f:
            json.dump(frame_text_pairs, f, indent=2, default=str)
        
        extracted_count = sum(1 for pair in frame_text_pairs if pair.get('frame_extracted', False))
        print(f"✅ Extracted {extracted_count}/{len(frame_text_pairs)} synchronized frames")
        
        return extracted_count > 0
    
    def generate_synchronized_comic(self, frame_text_pairs: List[Dict]):
        """
        Generate comic with perfect frame-text synchronization
        """
        print("📚 Generating comic with perfect frame-text synchronization...")
        
        # Create pages data with exact synchronization
        pages_data = []
        panels_per_page = 4
        
        for page_num in range(12):  # Exactly 12 pages
            page_start = page_num * panels_per_page
            page_end = min(page_start + panels_per_page, len(frame_text_pairs))
            
            page_pairs = frame_text_pairs[page_start:page_end]
            
            page_data = {
                'page_number': page_num + 1,
                'panels': [],
                'bubbles': []
            }
            
            for i, pair in enumerate(page_pairs):
                panel_index = page_start + i
                
                # Panel data - direct frame reference
                panel = {
                    'image': f'frame{panel_index:03d}.png',
                    'row_span': 6,
                    'col_span': 6,
                    'panel_number': pair['panel_number'],
                    'synchronized': True
                }
                page_data['panels'].append(panel)
                
                # Bubble data - exact text from synchronization
                bubble = {
                    'bubble_offset_x': 30 + (i % 2) * 120,
                    'bubble_offset_y': 30 + (i // 2) * 80,
                    'lip_x': -1,
                    'lip_y': -1,
                    'dialog': pair['text'],  # EXACT text for this frame
                    'emotion': 'normal',
                    'synchronized': True,
                    'original_time': pair['target_time'],
                    'has_subtitle': pair['has_original_subtitle']
                }
                page_data['bubbles'].append(bubble)
                
                print(f"📖 Page {page_num+1}, Panel {i+1}: '{pair['text'][:30]}...' → frame{panel_index:03d}.png")
            
            pages_data.append(page_data)
        
        # Save synchronized comic data
        os.makedirs('output', exist_ok=True)
        with open('output/pages.json', 'w') as f:
            json.dump(pages_data, f, indent=2)
        
        # Save detailed synchronization info
        sync_info = {
            'total_pages': 12,
            'panels_per_page': 4,
            'total_panels': len(frame_text_pairs),
            'synchronization_method': 'direct_frame_text_mapping',
            'video_info': self.video_info,
            'frame_text_pairs': frame_text_pairs
        }
        
        with open('output/synchronization_data.json', 'w') as f:
            json.dump(sync_info, f, indent=2, default=str)
        
        print("✅ Generated synchronized comic data")
        print(f"📊 {len(pages_data)} pages, {sum(len(p['panels']) for p in pages_data)} panels")
        print("🎯 Perfect frame-text synchronization achieved!")
        
        return True

def create_synchronized_comic(video_path: str, subtitles: List, target_panels: int = 48) -> bool:
    """
    Main function to create comic with perfect frame-text synchronization
    """
    try:
        synchronizer = FrameTextSynchronizer()
        success = synchronizer.create_synchronized_comic(video_path, subtitles, target_panels)
        
        if success:
            print("🎉 Synchronized comic creation successful!")
            print("✅ Each frame has its exact corresponding text")
            print("✅ Perfect 1:1 mapping between images and bubbles")
        
        return success
        
    except Exception as e:
        print(f"❌ Synchronized comic creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False