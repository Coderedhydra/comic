"""
Story Analyzer and Summarizer
Analyzes video content to create compelling comic summaries with emotion matching
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple
import srt
from datetime import timedelta
import re

# Try to import advanced NLP libraries
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available, using basic analysis")

class EmotionDetector:
    """Detect facial emotions in frames"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Emotion keywords for text analysis
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'laugh', 'smile', 'excited', 'wonderful', 'great', 'amazing', 'love', 'yes', 'haha', 'yay'],
            'sad': ['sad', 'cry', 'tear', 'sorry', 'miss', 'lonely', 'depressed', 'unhappy', 'grief', 'mourn'],
            'angry': ['angry', 'mad', 'furious', 'hate', 'annoyed', 'frustrated', 'rage', 'damn', 'hell', 'stupid'],
            'surprised': ['surprised', 'shock', 'wow', 'oh', 'what', 'really', 'seriously', 'unbelievable', 'amazing'],
            'fear': ['afraid', 'scared', 'fear', 'terrified', 'nervous', 'worry', 'anxious', 'panic', 'help'],
            'neutral': ['okay', 'fine', 'yes', 'no', 'maybe', 'think', 'know', 'understand']
        }
        
        # Try to load emotion detection model
        self.emotion_model = None
        try:
            if TRANSFORMERS_AVAILABLE:
                self.emotion_model = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
                print("✅ Emotion detection model loaded")
        except:
            print("⚠️ Using keyword-based emotion detection")
    
    def detect_facial_emotion(self, image_path: str) -> Dict[str, float]:
        """Detect emotion from facial expression"""
        img = cv2.imread(image_path)
        if img is None:
            return {'neutral': 1.0}
        
        # If we have the model, use it
        if self.emotion_model:
            try:
                from PIL import Image
                pil_img = Image.open(image_path)
                results = self.emotion_model(pil_img)
                
                emotions = {}
                for result in results:
                    emotion = result['label'].lower()
                    score = result['score']
                    emotions[emotion] = score
                
                return emotions
            except:
                pass
        
        # Fallback: Basic emotion detection using facial features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {'neutral': 1.0}
        
        # Analyze the largest face
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Simple heuristics based on facial features
        emotions = self._analyze_face_features(face_roi)
        return emotions
    
    def _analyze_face_features(self, face_roi):
        """Analyze facial features for emotion (basic heuristic)"""
        h, w = face_roi.shape
        
        # Divide face into regions
        upper_face = face_roi[0:int(h*0.5), :]  # Eyes and eyebrows
        lower_face = face_roi[int(h*0.5):, :]   # Mouth area
        
        # Calculate feature metrics
        upper_variance = np.var(upper_face)
        lower_variance = np.var(lower_face)
        
        # Simple emotion estimation
        emotions = {'neutral': 0.4}
        
        # High variance in lower face might indicate smile or frown
        if lower_variance > upper_variance * 1.5:
            emotions['happy'] = 0.6
        elif upper_variance > lower_variance * 1.5:
            emotions['surprised'] = 0.5
        else:
            emotions['neutral'] = 0.8
            
        return emotions
    
    def detect_text_emotion(self, text: str) -> Dict[str, float]:
        """Detect emotion from text/dialogue"""
        if not text:
            return {'neutral': 1.0}
        
        text_lower = text.lower()
        emotion_scores = {}
        
        # Check for emotion keywords
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                emotion_scores[emotion] = min(score * 0.3, 1.0)
        
        # Check for punctuation hints
        if '!' in text:
            emotion_scores['excited'] = emotion_scores.get('excited', 0) + 0.3
        if '?' in text:
            emotion_scores['surprised'] = emotion_scores.get('surprised', 0) + 0.2
        if '...' in text:
            emotion_scores['sad'] = emotion_scores.get('sad', 0) + 0.2
        
        # Normalize scores
        if emotion_scores:
            total = sum(emotion_scores.values())
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        else:
            emotion_scores = {'neutral': 1.0}
        
        return emotion_scores

class StoryAnalyzer:
    """Analyze story structure and extract key moments"""
    
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        
        # Story arc keywords
        self.story_elements = {
            'introduction': ['begin', 'start', 'once', 'first', 'meet', 'introduce', 'hello'],
            'conflict': ['but', 'however', 'problem', 'challenge', 'difficult', 'trouble', 'fight', 'argue'],
            'climax': ['finally', 'suddenly', 'then', 'biggest', 'most', 'intense', 'peak', 'critical'],
            'resolution': ['end', 'finally', 'resolve', 'solve', 'peace', 'happy', 'conclude', 'finish']
        }
        
        # Try to load summarization model
        self.summarizer = None
        try:
            if TRANSFORMERS_AVAILABLE:
                self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                print("✅ Story summarization model loaded")
        except:
            print("⚠️ Using rule-based summarization")
    
    def analyze_story(self, subtitles: List[srt.Subtitle], frames: List[str]) -> Dict:
        """Analyze the complete story structure"""
        
        # 1. Extract full story text
        full_text = ' '.join([sub.content for sub in subtitles])
        
        # 2. Identify story arc
        story_arc = self._identify_story_arc(subtitles)
        
        # 3. Extract key moments
        key_moments = self._extract_key_moments(subtitles, story_arc)
        
        # 4. Match emotions to moments
        emotional_moments = self._match_emotions(key_moments, frames)
        
        # 5. Create story summary
        summary = self._create_summary(full_text, key_moments)
        
        return {
            'story_arc': story_arc,
            'key_moments': key_moments,
            'emotional_moments': emotional_moments,
            'summary': summary,
            'total_duration': subtitles[-1].end if subtitles else timedelta(0)
        }
    
    def _identify_story_arc(self, subtitles: List[srt.Subtitle]) -> Dict:
        """Identify the story structure"""
        total_subs = len(subtitles)
        if total_subs == 0:
            return {}
        
        # Divide story into acts
        act1_end = int(total_subs * 0.25)  # First 25% - Introduction
        act2_end = int(total_subs * 0.75)  # Middle 50% - Development
        # Last 25% - Resolution
        
        story_arc = {
            'introduction': subtitles[:act1_end],
            'development': subtitles[act1_end:act2_end],
            'climax': subtitles[act2_end:act2_end + int(total_subs * 0.1)],
            'resolution': subtitles[act2_end:]
        }
        
        return story_arc
    
    def _extract_key_moments(self, subtitles: List[srt.Subtitle], story_arc: Dict) -> List[Dict]:
        """Extract 10-15 key moments from the story"""
        key_moments = []
        
        # Ensure we get moments from each story phase
        phases = ['introduction', 'development', 'climax', 'resolution']
        moments_per_phase = {
            'introduction': 2,
            'development': 6,
            'climax': 3,
            'resolution': 2
        }
        
        for phase in phases:
            if phase not in story_arc:
                continue
                
            phase_subs = story_arc[phase]
            if not phase_subs:
                continue
            
            # Select important moments from this phase
            num_moments = moments_per_phase.get(phase, 3)
            selected = self._select_important_subtitles(phase_subs, num_moments)
            
            for sub in selected:
                key_moments.append({
                    'subtitle': sub,
                    'phase': phase,
                    'timestamp': sub.start,
                    'text': sub.content,
                    'importance': self._calculate_importance(sub)
                })
        
        # Sort by timestamp
        key_moments.sort(key=lambda x: x['timestamp'])
        
        # Limit to 15 moments
        return key_moments[:15]
    
    def _select_important_subtitles(self, subtitles: List[srt.Subtitle], num: int) -> List[srt.Subtitle]:
        """Select the most important subtitles from a list"""
        if len(subtitles) <= num:
            return subtitles
        
        # Score each subtitle
        scored_subs = []
        for sub in subtitles:
            score = self._calculate_importance(sub)
            scored_subs.append((sub, score))
        
        # Sort by score and select top N
        scored_subs.sort(key=lambda x: x[1], reverse=True)
        selected = [sub for sub, score in scored_subs[:num]]
        
        # Sort back by time
        selected.sort(key=lambda x: x.start)
        
        return selected
    
    def _calculate_importance(self, subtitle: srt.Subtitle) -> float:
        """Calculate importance score for a subtitle"""
        text = subtitle.content.lower()
        score = 1.0
        
        # Length bonus (longer usually more important)
        score += len(text.split()) * 0.1
        
        # Punctuation bonus
        if '!' in text:
            score += 0.5
        if '?' in text:
            score += 0.3
        
        # Emotion words bonus
        emotion_words = ['love', 'hate', 'fear', 'happy', 'sad', 'angry', 'surprise']
        for word in emotion_words:
            if word in text:
                score += 0.5
        
        # Action words bonus
        action_words = ['fight', 'run', 'escape', 'save', 'help', 'stop', 'go']
        for word in action_words:
            if word in text:
                score += 0.4
        
        # Story element bonus
        for element, keywords in self.story_elements.items():
            for keyword in keywords:
                if keyword in text:
                    score += 0.3
        
        return score
    
    def _match_emotions(self, key_moments: List[Dict], frames: List[str]) -> List[Dict]:
        """Match emotions between text and facial expressions"""
        emotional_moments = []
        
        for moment in key_moments:
            # Get text emotion
            text_emotions = self.emotion_detector.detect_text_emotion(moment['text'])
            
            # Find the best matching frame based on timestamp
            best_frame = self._find_best_frame(moment['timestamp'], frames)
            
            if best_frame:
                # Get facial emotion
                facial_emotions = self.emotion_detector.detect_facial_emotion(best_frame)
                
                # Combine emotions
                combined_emotions = self._combine_emotions(text_emotions, facial_emotions)
                
                emotional_moments.append({
                    'moment': moment,
                    'frame': best_frame,
                    'text_emotions': text_emotions,
                    'facial_emotions': facial_emotions,
                    'combined_emotions': combined_emotions,
                    'dominant_emotion': max(combined_emotions.items(), key=lambda x: x[1])[0]
                })
            else:
                emotional_moments.append({
                    'moment': moment,
                    'frame': None,
                    'text_emotions': text_emotions,
                    'facial_emotions': {'neutral': 1.0},
                    'combined_emotions': text_emotions,
                    'dominant_emotion': max(text_emotions.items(), key=lambda x: x[1])[0]
                })
        
        return emotional_moments
    
    def _find_best_frame(self, timestamp: timedelta, frames: List[str]) -> str:
        """Find the frame closest to the given timestamp"""
        # This is a simplified version - in real implementation,
        # you would map frame numbers to video timestamps
        
        # For now, return a frame based on position
        total_frames = len(frames)
        if total_frames == 0:
            return None
        
        # Simple mapping (would be more sophisticated in practice)
        frame_index = int(timestamp.total_seconds() * 2) % total_frames
        return frames[frame_index] if frame_index < total_frames else frames[-1]
    
    def _combine_emotions(self, text_emotions: Dict[str, float], 
                         facial_emotions: Dict[str, float]) -> Dict[str, float]:
        """Combine text and facial emotions"""
        combined = {}
        
        # Weight: 60% facial, 40% text (faces are more reliable)
        all_emotions = set(text_emotions.keys()) | set(facial_emotions.keys())
        
        for emotion in all_emotions:
            text_score = text_emotions.get(emotion, 0)
            facial_score = facial_emotions.get(emotion, 0)
            combined[emotion] = (facial_score * 0.6) + (text_score * 0.4)
        
        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v/total for k, v in combined.items()}
        
        return combined
    
    def _create_summary(self, full_text: str, key_moments: List[Dict]) -> str:
        """Create a concise summary of the story"""
        if self.summarizer and len(full_text) > 100:
            try:
                # Use AI summarization
                summary = self.summarizer(full_text, max_length=130, min_length=30, do_sample=False)
                return summary[0]['summary_text']
            except:
                pass
        
        # Fallback: Create summary from key moments
        summary_parts = []
        for moment in key_moments[:5]:  # Use first 5 key moments
            summary_parts.append(moment['text'])
        
        return ' '.join(summary_parts)

class SmartComicGenerator:
    """Generate comics with emotion-matched panels and story summarization"""
    
    def __init__(self):
        self.story_analyzer = StoryAnalyzer()
        self.panel_target = 12  # Target number of panels
    
    def generate_smart_comic(self, video_path: str, output_dir: str = 'output'):
        """Generate a smart comic with emotion matching and story summary"""
        print("🎬 Generating Smart Comic with Emotion Matching...")
        
        # 1. Load subtitles
        subtitles = self._load_subtitles()
        
        # 2. Get all available frames
        frames = self._get_frames()
        
        # 3. Analyze story
        print("📖 Analyzing story structure...")
        story_analysis = self.story_analyzer.analyze_story(subtitles, frames)
        
        # 4. Select best panels with emotion matching
        print("🎭 Matching emotions and selecting key moments...")
        comic_panels = self._select_comic_panels(story_analysis)
        
        # 5. Generate comic layout
        print("📐 Creating comic layout...")
        comic_data = self._create_comic_layout(comic_panels)
        
        # 6. Save comic
        self._save_comic(comic_data, output_dir)
        
        print(f"✅ Smart comic generated with {len(comic_panels)} panels!")
        print(f"📝 Story summary: {story_analysis['summary'][:100]}...")
        
        return comic_data
    
    def _load_subtitles(self) -> List[srt.Subtitle]:
        """Load subtitles from file"""
        if os.path.exists('test1.srt'):
            with open('test1.srt', 'r') as f:
                return list(srt.parse(f.read()))
        return []
    
    def _get_frames(self) -> List[str]:
        """Get all available frames"""
        frames_dir = 'frames/final' if os.path.exists('frames/final') else 'frames'
        frames = []
        
        if os.path.exists(frames_dir):
            frames = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir)) 
                     if f.endswith('.png')]
        
        return frames
    
    def _select_comic_panels(self, story_analysis: Dict) -> List[Dict]:
        """Select the best panels for the comic"""
        emotional_moments = story_analysis['emotional_moments']
        
        # Filter to get diverse emotions and story coverage
        selected_panels = []
        used_emotions = set()
        
        # Ensure we get introduction, climax, and resolution
        phases_needed = ['introduction', 'development', 'climax', 'resolution']
        
        for phase in phases_needed:
            phase_moments = [m for m in emotional_moments 
                           if m['moment']['phase'] == phase]
            
            if phase_moments:
                # Select moments with different emotions
                for moment in phase_moments:
                    emotion = moment['dominant_emotion']
                    
                    # Prioritize new emotions or important moments
                    if emotion not in used_emotions or moment['moment']['importance'] > 3:
                        selected_panels.append(moment)
                        used_emotions.add(emotion)
                        
                        if len(selected_panels) >= self.panel_target:
                            break
        
        # If we need more panels, add based on importance
        if len(selected_panels) < self.panel_target:
            remaining = [m for m in emotional_moments if m not in selected_panels]
            remaining.sort(key=lambda x: x['moment']['importance'], reverse=True)
            selected_panels.extend(remaining[:self.panel_target - len(selected_panels)])
        
        # Sort by timestamp
        selected_panels.sort(key=lambda x: x['moment']['timestamp'])
        
        # Limit to target
        return selected_panels[:self.panel_target]
    
    def _create_comic_layout(self, comic_panels: List[Dict]) -> Dict:
        """Create comic layout with panels and emotion-matched bubbles"""
        pages = []
        panels_per_page = 4
        
        for i in range(0, len(comic_panels), panels_per_page):
            page_panels = comic_panels[i:i+panels_per_page]
            
            page = {
                'width': 800,
                'height': 600,
                'panels': [],
                'bubbles': []
            }
            
            # 2x2 grid positions
            positions = [
                (10, 10, 380, 280),
                (410, 10, 380, 280),
                (10, 310, 380, 280),
                (410, 310, 380, 280)
            ]
            
            for j, panel_data in enumerate(page_panels):
                if j >= 4:
                    break
                    
                x, y, w, h = positions[j]
                
                # Add panel
                page['panels'].append({
                    'x': x, 'y': y,
                    'width': w, 'height': h,
                    'image': panel_data['frame'] or '/frames/frame000.png',
                    'emotion': panel_data['dominant_emotion']
                })
                
                # Add emotion-styled bubble
                bubble_style = self._get_bubble_style(panel_data['dominant_emotion'])
                
                page['bubbles'].append({
                    'id': f'bubble_{i+j}',
                    'x': x + 20,
                    'y': y + 20,
                    'width': 180,
                    'height': 80,
                    'text': panel_data['moment']['text'],
                    'emotion': panel_data['dominant_emotion'],
                    'style': bubble_style
                })
            
            pages.append(page)
        
        return {
            'pages': pages,
            'summary': comic_panels[0]['moment'].get('summary', ''),
            'total_panels': len(comic_panels)
        }
    
    def _get_bubble_style(self, emotion: str) -> Dict:
        """Get bubble style based on emotion"""
        styles = {
            'happy': {
                'border_color': '#4CAF50',
                'background': '#E8F5E9',
                'font_size': 16,
                'border_width': 3
            },
            'sad': {
                'border_color': '#2196F3',
                'background': '#E3F2FD',
                'font_size': 14,
                'border_width': 2
            },
            'angry': {
                'border_color': '#F44336',
                'background': '#FFEBEE',
                'font_size': 18,
                'border_width': 4,
                'jagged': True
            },
            'surprised': {
                'border_color': '#FF9800',
                'background': '#FFF3E0',
                'font_size': 16,
                'border_width': 3,
                'exclamation': True
            },
            'fear': {
                'border_color': '#9C27B0',
                'background': '#F3E5F5',
                'font_size': 14,
                'border_width': 2,
                'wavy': True
            },
            'neutral': {
                'border_color': '#333',
                'background': '#FFF',
                'font_size': 14,
                'border_width': 2
            }
        }
        
        return styles.get(emotion, styles['neutral'])
    
    def _save_comic(self, comic_data: Dict, output_dir: str):
        """Save comic data and generate HTML"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON data
        with open(os.path.join(output_dir, 'smart_comic.json'), 'w') as f:
            json.dump(comic_data, f, indent=2, default=str)
        
        # Generate HTML
        html = self._generate_html(comic_data)
        with open(os.path.join(output_dir, 'smart_comic.html'), 'w') as f:
            f.write(html)
    
    def _generate_html(self, comic_data: Dict) -> str:
        """Generate HTML for the smart comic"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Smart Comic - Emotion Matched</title>
    <style>
        body { margin: 0; padding: 20px; background: #f0f0f0; font-family: Arial, sans-serif; }
        .comic-page { position: relative; background: white; margin: 20px auto; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .comic-panel { position: absolute; border: 3px solid #333; overflow: hidden; }
        .comic-panel img { width: 100%; height: 100%; object-fit: cover; }
        .speech-bubble { position: absolute; border-radius: 20px; padding: 15px; font-family: 'Comic Sans MS', cursive; font-weight: bold; text-align: center; z-index: 10; }
        .emotion-happy { animation: bounce 2s infinite; }
        .emotion-sad { opacity: 0.8; }
        .emotion-angry { animation: shake 0.5s infinite; }
        .emotion-surprised { transform: scale(1.1); }
        @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
        @keyframes shake { 0%, 100% { transform: translateX(0); } 25% { transform: translateX(-2px); } 75% { transform: translateX(2px); } }
    </style>
</head>
<body>
"""
        
        for page in comic_data['pages']:
            html += f'<div class="comic-page" style="width:{page["width"]}px;height:{page["height"]}px;">\n'
            
            for panel in page['panels']:
                html += f'<div class="comic-panel" style="left:{panel["x"]}px;top:{panel["y"]}px;width:{panel["width"]}px;height:{panel["height"]}px;">'
                html += f'<img src="{panel["image"]}" alt="Panel">'
                html += '</div>\n'
            
            for bubble in page['bubbles']:
                style = bubble.get('style', {})
                emotion_class = f"emotion-{bubble.get('emotion', 'neutral')}"
                
                html += f'<div class="speech-bubble {emotion_class}" style="'
                html += f'left:{bubble["x"]}px;top:{bubble["y"]}px;'
                html += f'width:{bubble["width"]}px;height:{bubble["height"]}px;'
                html += f'border: {style.get("border_width", 2)}px solid {style.get("border_color", "#333")};'
                html += f'background: {style.get("background", "#FFF")};'
                html += f'font-size: {style.get("font_size", 14)}px;">'
                html += bubble["text"]
                html += '</div>\n'
            
            html += '</div>\n'
        
        html += """
</body>
</html>"""
        
        return html

# Quick test function
def test_smart_comic(video_path='video/sample.mp4'):
    """Test the smart comic generation"""
    generator = SmartComicGenerator()
    comic_data = generator.generate_smart_comic(video_path)
    print("✅ Smart comic generated successfully!")
    return comic_data

if __name__ == "__main__":
    test_smart_comic()