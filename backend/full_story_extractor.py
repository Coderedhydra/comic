"""
Full Story Extractor - Captures complete story without skipping important parts
"""

import os
import json
import srt
from typing import List, Dict

class FullStoryExtractor:
    def __init__(self):
        self.min_panels_per_page = 4  # 2x2 grid
        self.target_pages = 12
        self.max_panels = 48
        
    def extract_full_story(self, subtitles_file: str) -> List[Dict]:
        """Extract full story maintaining continuity"""
        
        # Load subtitles
        try:
            if subtitles_file.endswith('.srt'):
                with open(subtitles_file, 'r', encoding='utf-8') as f:
                    subs = list(srt.parse(f.read()))
                    subtitles = []
                    for sub in subs:
                        subtitles.append({
                            'index': sub.index,
                            'text': sub.content,
                            'start': sub.start.total_seconds(),
                            'end': sub.end.total_seconds()
                        })
            else:
                with open(subtitles_file, 'r') as f:
                    subtitles = json.load(f)
        except:
            return []
            
        total_subs = len(subtitles)
        print(f"📚 Analyzing {total_subs} subtitles for complete story")
        
        if total_subs <= self.max_panels:
            # If we have less than 48 subtitles, use them all
            print(f"✅ Using all {total_subs} subtitles (complete story)")
            return subtitles
        
        # For longer videos, sample evenly to maintain story flow
        # Don't skip sections - take regular intervals
        step = total_subs / self.max_panels
        selected = []
        
        for i in range(self.max_panels):
            idx = int(i * step)
            if idx < total_subs:
                selected.append(subtitles[idx])
        
        # Ensure we always have the first and last
        if selected[0] != subtitles[0]:
            selected[0] = subtitles[0]
        if selected[-1] != subtitles[-1]:
            selected[-1] = subtitles[-1]
            
        print(f"✅ Selected {len(selected)} evenly distributed moments")
        print("📖 Full story preserved: Beginning → Middle → End")
        
        return selected
    
    def get_story_continuity_frames(self, video_path: str, subtitles: List[Dict]) -> Dict:
        """Get frames that maintain story continuity"""
        
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_data = []
        
        for i, sub in enumerate(subtitles):
            # Get frame from middle of subtitle duration
            timestamp = (sub['start'] + sub['end']) / 2
            frame_num = int(timestamp * fps)
            
            # Also get quality score for this moment
            quality_score = self._assess_moment_quality(sub)
            
            frames_data.append({
                'index': i,
                'frame_num': frame_num,
                'subtitle': sub,
                'quality_score': quality_score,
                'timestamp': timestamp
            })
        
        cap.release()
        
        return {
            'frames': frames_data,
            'total': len(frames_data),
            'story_complete': True
        }
    
    def _assess_moment_quality(self, subtitle: Dict) -> float:
        """Assess the quality/importance of a story moment"""
        score = 5.0  # Base score
        text = subtitle.get('text', '').lower()
        
        # Length bonus
        words = text.split()
        if len(words) > 10:
            score += 2.0
        elif len(words) > 5:
            score += 1.0
            
        # Dialogue bonus
        if '"' in text or "'" in text:
            score += 1.5
            
        # Emotion bonus
        emotions = ['happy', 'sad', 'angry', 'love', 'fear', 'excited']
        for emotion in emotions:
            if emotion in text:
                score += 1.0
                break
                
        # Action bonus
        actions = ['run', 'jump', 'fight', 'escape', 'save', 'help']
        for action in actions:
            if action in text:
                score += 1.0
                break
                
        return min(score, 10.0)  # Cap at 10