"""
Simple, clean comic generator that:
1. Selects ONLY 12 meaningful story moments
2. Preserves original image quality and colors
3. Uses proper grid layouts (3x4 for 12 panels)
"""

import os
import cv2
import json
import srt
import numpy as np
from typing import List, Dict

class SimpleComicGenerator:
    def __init__(self):
        self.target_panels = 12
        self.frames_dir = 'frames/final'
        self.output_dir = 'output'
        
    def generate_meaningful_comic(self, video_path: str) -> bool:
        """Generate comic with only meaningful story moments"""
        try:
            print("🎬 Starting Simple Comic Generation...")
            print(f"📊 Target: {self.target_panels} meaningful panels")
            
            # 1. Get subtitles
            subtitles = self._load_subtitles()
            if not subtitles:
                print("❌ No subtitles found")
                return False
                
            print(f"📝 Found {len(subtitles)} total subtitles")
            
            # 2. Select ONLY meaningful moments
            meaningful_moments = self._select_meaningful_moments(subtitles)
            print(f"✅ Selected {len(meaningful_moments)} key story moments")
            
            # 3. Extract frames for these moments only
            self._extract_meaningful_frames(video_path, meaningful_moments)
            
            # 4. Create simple grid layout (NO styling, preserve colors)
            self._create_comic_pages()
            
            print("✅ Comic generation complete!")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def _load_subtitles(self) -> List[Dict]:
        """Load subtitles from SRT file"""
        try:
            with open('test1.srt', 'r', encoding='utf-8') as f:
                subs = list(srt.parse(f.read()))
                
            # Convert to dict format
            subtitle_list = []
            for sub in subs:
                subtitle_list.append({
                    'index': sub.index,
                    'text': sub.content,
                    'start': sub.start.total_seconds(),
                    'end': sub.end.total_seconds()
                })
            return subtitle_list
        except:
            return []
    
    def _select_meaningful_moments(self, subtitles: List[Dict]) -> List[Dict]:
        """Select ONLY the most meaningful story moments"""
        
        # Score each subtitle
        scored_subs = []
        total = len(subtitles)
        
        for i, sub in enumerate(subtitles):
            score = 0
            text = sub['text'].lower()
            position = i / total
            
            # 1. Story position scoring
            if position < 0.1:  # Introduction
                score += 5
            elif position > 0.9:  # Conclusion
                score += 5
            elif 0.45 < position < 0.55:  # Climax area
                score += 4
                
            # 2. Content importance
            important_words = [
                'but', 'however', 'suddenly', 'finally', 'then',
                'help', 'save', 'fight', 'love', 'hate', 'die',
                'win', 'lose', 'find', 'discover', 'realize',
                'important', 'must', 'need', 'want'
            ]
            
            for word in important_words:
                if word in text:
                    score += 3
                    
            # 3. Emotional content
            if '!' in text:
                score += 2
            if '?' in text:
                score += 1
                
            # 4. Length (longer = more important)
            if len(text.split()) > 10:
                score += 2
            elif len(text.split()) > 5:
                score += 1
                
            scored_subs.append((score, i, sub))
        
        # Sort by score
        scored_subs.sort(key=lambda x: x[0], reverse=True)
        
        # Select top moments with good distribution
        selected = []
        selected_indices = set()
        
        # Ensure we get intro and conclusion
        if subtitles:
            selected.append(subtitles[0])  # First
            selected_indices.add(0)
            if len(subtitles) > 1:
                selected.append(subtitles[-1])  # Last
                selected_indices.add(len(subtitles) - 1)
        
        # Add high-scoring moments with spacing
        min_spacing = max(1, total // (self.target_panels * 2))
        
        for score, idx, sub in scored_subs:
            if len(selected) >= self.target_panels:
                break
                
            # Check spacing
            too_close = False
            for sel_idx in selected_indices:
                if abs(idx - sel_idx) < min_spacing:
                    too_close = True
                    break
                    
            if not too_close and idx not in selected_indices:
                selected.append(sub)
                selected_indices.add(idx)
        
        # Sort by time
        selected.sort(key=lambda x: x['start'])
        
        # Limit to target
        return selected[:self.target_panels]
    
    def _extract_meaningful_frames(self, video_path: str, moments: List[Dict]):
        """Extract frames ONLY for meaningful moments"""
        
        # Clear frames directory
        os.makedirs(self.frames_dir, exist_ok=True)
        for f in os.listdir(self.frames_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(self.frames_dir, f))
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"🎥 Extracting {len(moments)} frames...")
        
        for i, moment in enumerate(moments):
            # Get frame at subtitle midpoint
            timestamp = (moment['start'] + moment['end']) / 2
            frame_num = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Save frame WITHOUT any processing (preserve quality)
                output_path = os.path.join(self.frames_dir, f'frame{i:03d}.png')
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                print(f"  ✓ Frame {i+1}/{len(moments)}: {moment['text'][:50]}...")
            else:
                print(f"  ✗ Failed to extract frame {i+1}")
        
        cap.release()
        print(f"✅ Extracted {len(moments)} frames")
    
    def _create_comic_pages(self):
        """Create comic pages with proper grid layout"""
        
        frames = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        num_frames = len(frames)
        
        if num_frames == 0:
            print("❌ No frames to create comic")
            return
        
        print(f"📄 Creating comic with {num_frames} panels...")
        
        # Determine layout
        if num_frames <= 6:
            layout = "2x3"  # 2 rows, 3 columns
            rows, cols = 2, 3
        elif num_frames <= 9:
            layout = "3x3"  # 3 rows, 3 columns
            rows, cols = 3, 3
        elif num_frames <= 12:
            layout = "3x4"  # 3 rows, 4 columns
            rows, cols = 3, 4
        else:
            layout = "4x4"  # 4 rows, 4 columns
            rows, cols = 4, 4
        
        print(f"📐 Using {layout} grid layout")
        
        # Save comic data
        comic_data = {
            'frames': frames,
            'layout': layout,
            'rows': rows,
            'cols': cols,
            'total_panels': num_frames
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'comic_data.json'), 'w') as f:
            json.dump(comic_data, f, indent=2)
        
        # Create simple HTML viewer
        self._create_html_viewer(frames, rows, cols)
    
    def _create_html_viewer(self, frames: List[str], rows: int, cols: int):
        """Create simple HTML viewer for the comic"""
        
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Story Comic - 12 Key Moments</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #f0f0f0;
            font-family: Arial, sans-serif;
        }
        .comic-container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .comic-grid {
            display: grid;
            grid-template-columns: repeat(''' + str(cols) + ''', 1fr);
            grid-template-rows: repeat(''' + str(rows) + ''', 1fr);
            gap: 10px;
            width: 100%;
        }
        .panel {
            position: relative;
            border: 2px solid #333;
            overflow: hidden;
            background: #fff;
        }
        .panel img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .panel-number {
            position: absolute;
            top: 5px;
            left: 5px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
        }
        .info {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="comic-container">
        <h1>📚 Story Comic - Key Moments</h1>
        <div class="info">''' + str(len(frames)) + ''' panels showing the most important story moments</div>
        <div class="comic-grid">
'''
        
        for i, frame in enumerate(frames):
            html += f'''
            <div class="panel">
                <div class="panel-number">{i+1}</div>
                <img src="../frames/final/{frame}" alt="Panel {i+1}">
            </div>
'''
        
        html += '''
        </div>
    </div>
</body>
</html>'''
        
        output_path = os.path.join(self.output_dir, 'comic_simple.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Comic viewer saved to: {output_path}")