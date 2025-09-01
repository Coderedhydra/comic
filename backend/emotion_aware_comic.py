"""
Emotion-Aware Comic Generation
Creates comics that match facial expressions with dialogue emotions
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
import srt
from datetime import timedelta

class FacialExpressionAnalyzer:
    """Analyze facial expressions in frames"""
    
    def __init__(self):
        # Load face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
    def analyze_expression(self, image_path: str) -> Dict[str, float]:
        """Analyze facial expression in an image"""
        img = cv2.imread(image_path)
        if img is None:
            return self._default_expression()
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return self._default_expression()
        
        # Analyze the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect features
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5)
        smiles = self.smile_cascade.detectMultiScale(face_roi, 1.8, 20)
        
        # Analyze expression based on features
        expression = self._analyze_features(face_roi, eyes, smiles)
        
        # Add intensity analysis
        expression['intensity'] = self._analyze_intensity(face_roi)
        
        return expression
    
    def _analyze_features(self, face_roi, eyes, smiles) -> Dict[str, float]:
        """Analyze facial features to determine expression"""
        expression = {
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprised': 0.0,
            'neutral': 0.5
        }
        
        # Smile detection
        if len(smiles) > 0:
            expression['happy'] = 0.7
            expression['neutral'] = 0.3
        
        # Eye analysis
        if len(eyes) >= 2:
            # Both eyes visible - analyze eye region
            eye_region = face_roi[:face_roi.shape[0]//2, :]
            eye_variance = np.var(eye_region)
            
            if eye_variance > 1000:  # Wide eyes
                expression['surprised'] = 0.6
            elif eye_variance < 500:  # Squinted eyes
                expression['angry'] = 0.4
        elif len(eyes) < 2:
            # Eyes not clearly visible - might be closed or squinted
            expression['sad'] = 0.3
            expression['angry'] = 0.3
        
        # Normalize scores
        total = sum(expression.values())
        if total > 0:
            expression = {k: v/total for k, v in expression.items()}
            
        return expression
    
    def _analyze_intensity(self, face_roi) -> float:
        """Analyze expression intensity"""
        # Calculate contrast and edge density
        edges = cv2.Canny(face_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Higher edge density often means more intense expression
        intensity = min(edge_density * 5, 1.0)
        return intensity
    
    def _default_expression(self) -> Dict[str, float]:
        """Default expression when no face detected"""
        return {
            'neutral': 1.0,
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprised': 0.0,
            'intensity': 0.5
        }

class DialogueEmotionAnalyzer:
    """Analyze emotions in dialogue text"""
    
    def __init__(self):
        # Emotion lexicons
        self.emotion_words = {
            'happy': {
                'words': ['happy', 'joy', 'love', 'great', 'wonderful', 'amazing', 'fantastic', 'excellent', 'beautiful', 'laugh', 'smile', 'fun'],
                'weight': 1.0
            },
            'sad': {
                'words': ['sad', 'cry', 'tear', 'sorry', 'miss', 'lonely', 'depressed', 'hurt', 'pain', 'loss', 'grief'],
                'weight': 1.0
            },
            'angry': {
                'words': ['angry', 'mad', 'furious', 'hate', 'stupid', 'idiot', 'damn', 'hell', 'rage', 'annoyed'],
                'weight': 1.2
            },
            'surprised': {
                'words': ['wow', 'oh', 'what', 'really', 'seriously', 'unbelievable', 'amazing', 'shocked', 'surprised'],
                'weight': 0.8
            },
            'fear': {
                'words': ['afraid', 'scared', 'fear', 'terrified', 'nervous', 'worry', 'panic', 'help', 'danger'],
                'weight': 1.0
            }
        }
        
        # Punctuation patterns
        self.punctuation_emotions = {
            '!': {'surprised': 0.3, 'happy': 0.2, 'angry': 0.2},
            '?': {'surprised': 0.4, 'confused': 0.3},
            '...': {'sad': 0.3, 'thoughtful': 0.3},
            '?!': {'surprised': 0.6},
            '!!!': {'angry': 0.4, 'excited': 0.4}
        }
    
    def analyze_dialogue(self, text: str) -> Dict[str, float]:
        """Analyze emotion in dialogue text"""
        if not text:
            return {'neutral': 1.0}
        
        text_lower = text.lower()
        emotions = {'neutral': 0.2}  # Base neutral score
        
        # Word-based analysis
        for emotion, data in self.emotion_words.items():
            score = 0
            for word in data['words']:
                if word in text_lower:
                    score += data['weight']
            
            if score > 0:
                emotions[emotion] = score
        
        # Punctuation analysis
        for pattern, emotion_scores in self.punctuation_emotions.items():
            if pattern in text:
                for emotion, score in emotion_scores.items():
                    emotions[emotion] = emotions.get(emotion, 0) + score
        
        # Intensity based on caps and punctuation
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.5:
            emotions['intensity'] = 0.8
        else:
            emotions['intensity'] = 0.5
        
        # Normalize
        emotion_sum = sum(v for k, v in emotions.items() if k != 'intensity')
        if emotion_sum > 0:
            for k in emotions:
                if k != 'intensity':
                    emotions[k] = emotions[k] / emotion_sum
        
        return emotions

class StoryCondenser:
    """Condense long stories into key moments"""
    
    def __init__(self):
        self.min_panels = 10
        self.max_panels = 15
        
    def identify_key_moments(self, subtitles: List[srt.Subtitle]) -> List[int]:
        """Identify indices of key story moments"""
        if len(subtitles) <= self.max_panels:
            return list(range(len(subtitles)))
        
        key_indices = []
        
        # 1. Always include first and last (introduction and conclusion)
        key_indices.extend([0, len(subtitles) - 1])
        
        # 2. Identify turning points
        turning_points = self._find_turning_points(subtitles)
        key_indices.extend(turning_points)
        
        # 3. Find emotional peaks
        emotional_peaks = self._find_emotional_peaks(subtitles)
        key_indices.extend(emotional_peaks)
        
        # 4. Find action moments
        action_moments = self._find_action_moments(subtitles)
        key_indices.extend(action_moments)
        
        # Remove duplicates and sort
        key_indices = sorted(list(set(key_indices)))
        
        # 5. If too many, select most important
        if len(key_indices) > self.max_panels:
            key_indices = self._select_most_important(subtitles, key_indices)
        
        # 6. If too few, add transitional moments
        if len(key_indices) < self.min_panels:
            key_indices = self._add_transitions(subtitles, key_indices)
        
        return sorted(key_indices)[:self.max_panels]
    
    def _find_turning_points(self, subtitles: List[srt.Subtitle]) -> List[int]:
        """Find story turning points"""
        turning_words = ['but', 'however', 'suddenly', 'then', 'meanwhile', 'later', 'finally']
        indices = []
        
        for i, sub in enumerate(subtitles):
            text_lower = sub.content.lower()
            if any(word in text_lower for word in turning_words):
                indices.append(i)
        
        return indices
    
    def _find_emotional_peaks(self, subtitles: List[srt.Subtitle]) -> List[int]:
        """Find emotional peaks in dialogue"""
        analyzer = DialogueEmotionAnalyzer()
        emotion_scores = []
        
        for i, sub in enumerate(subtitles):
            emotions = analyzer.analyze_dialogue(sub.content)
            # Calculate emotional intensity
            intensity = max(v for k, v in emotions.items() if k != 'neutral')
            emotion_scores.append((i, intensity))
        
        # Sort by intensity and take top moments
        emotion_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, score in emotion_scores[:5] if score > 0.5]
    
    def _find_action_moments(self, subtitles: List[srt.Subtitle]) -> List[int]:
        """Find action moments"""
        action_words = ['run', 'fight', 'escape', 'attack', 'save', 'help', 'stop', 'go', 'move', 'quick']
        indices = []
        
        for i, sub in enumerate(subtitles):
            text_lower = sub.content.lower()
            if any(word in text_lower for word in action_words):
                indices.append(i)
        
        return indices
    
    def _select_most_important(self, subtitles: List[srt.Subtitle], indices: List[int]) -> List[int]:
        """Select most important moments from candidates"""
        scored_indices = []
        
        for idx in indices:
            score = self._calculate_importance_score(subtitles[idx], idx, len(subtitles))
            scored_indices.append((idx, score))
        
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, score in scored_indices[:self.max_panels]]
    
    def _calculate_importance_score(self, subtitle: srt.Subtitle, index: int, total: int) -> float:
        """Calculate importance score for a subtitle"""
        score = 1.0
        
        # Position in story (beginning and end are important)
        position_ratio = index / total
        if position_ratio < 0.1 or position_ratio > 0.9:
            score += 0.5
        elif 0.4 < position_ratio < 0.6:  # Middle (potential climax)
            score += 0.3
        
        # Length (longer usually more important)
        word_count = len(subtitle.content.split())
        score += min(word_count * 0.1, 0.5)
        
        # Punctuation (excitement)
        if '!' in subtitle.content:
            score += 0.3
        if '?' in subtitle.content:
            score += 0.2
        
        return score
    
    def _add_transitions(self, subtitles: List[srt.Subtitle], current_indices: List[int]) -> List[int]:
        """Add transitional moments between key points"""
        new_indices = list(current_indices)
        
        # Find largest gaps
        gaps = []
        for i in range(len(current_indices) - 1):
            gap_size = current_indices[i+1] - current_indices[i]
            if gap_size > 2:
                gaps.append((current_indices[i], current_indices[i+1], gap_size))
        
        # Sort by gap size
        gaps.sort(key=lambda x: x[2], reverse=True)
        
        # Add midpoints of largest gaps
        for start, end, size in gaps:
            if len(new_indices) >= self.min_panels:
                break
            midpoint = (start + end) // 2
            new_indices.append(midpoint)
        
        return sorted(new_indices)

class EmotionAwareComicGenerator:
    """Generate comics with emotion-aware panel selection"""
    
    def __init__(self):
        self.face_analyzer = FacialExpressionAnalyzer()
        self.dialogue_analyzer = DialogueEmotionAnalyzer()
        self.story_condenser = StoryCondenser()
        
    def generate_emotion_comic(self, video_path: str, max_panels: int = 12) -> Dict:
        """Generate comic with emotion-matched panels"""
        print("🎭 Generating Emotion-Aware Comic...")
        
        # 1. Load subtitles and frames
        subtitles = self._load_subtitles()
        all_frames = self._get_all_frames()
        
        if not subtitles or not all_frames:
            print("❌ Missing subtitles or frames")
            return None
        
        # 2. Identify key story moments
        print("📖 Identifying key story moments...")
        key_indices = self.story_condenser.identify_key_moments(subtitles)
        print(f"  Found {len(key_indices)} key moments")
        
        # 3. Match emotions for each moment
        print("🎭 Matching facial expressions with dialogue...")
        matched_panels = []
        
        for idx in key_indices:
            subtitle = subtitles[idx]
            
            # Analyze dialogue emotion
            text_emotions = self.dialogue_analyzer.analyze_dialogue(subtitle.content)
            
            # Find best matching frame
            best_frame = self._find_best_emotion_match(
                subtitle, text_emotions, all_frames, idx, len(subtitles)
            )
            
            matched_panels.append({
                'subtitle': subtitle,
                'frame': best_frame['path'],
                'text_emotions': text_emotions,
                'face_emotions': best_frame['emotions'],
                'match_score': best_frame['score'],
                'index': idx
            })
        
        # 4. Create comic layout
        print("📐 Creating emotion-aware layout...")
        comic_data = self._create_emotion_layout(matched_panels)
        
        # 5. Save comic
        self._save_emotion_comic(comic_data)
        
        print(f"✅ Emotion-aware comic created with {len(matched_panels)} panels!")
        return comic_data
    
    def _find_best_emotion_match(self, subtitle: srt.Subtitle, text_emotions: Dict,
                                 frames: List[str], sub_index: int, total_subs: int) -> Dict:
        """Find frame with best emotion match"""
        
        # Calculate approximate frame range for this subtitle
        frame_ratio = sub_index / total_subs
        center_frame = int(frame_ratio * len(frames))
        
        # Search window (look at nearby frames)
        search_range = 5
        start = max(0, center_frame - search_range)
        end = min(len(frames), center_frame + search_range + 1)
        
        best_match = {
            'path': frames[center_frame] if center_frame < len(frames) else frames[-1],
            'emotions': {'neutral': 1.0},
            'score': 0
        }
        
        # Find best matching frame
        for i in range(start, end):
            if i >= len(frames):
                break
                
            # Analyze facial expression
            face_emotions = self.face_analyzer.analyze_expression(frames[i])
            
            # Calculate match score
            score = self._calculate_emotion_match_score(text_emotions, face_emotions)
            
            if score > best_match['score']:
                best_match = {
                    'path': frames[i],
                    'emotions': face_emotions,
                    'score': score
                }
        
        return best_match
    
    def _calculate_emotion_match_score(self, text_emotions: Dict, face_emotions: Dict) -> float:
        """Calculate how well emotions match"""
        score = 0
        
        # Compare each emotion
        emotions = set(text_emotions.keys()) | set(face_emotions.keys())
        for emotion in emotions:
            if emotion == 'intensity':
                continue
                
            text_score = text_emotions.get(emotion, 0)
            face_score = face_emotions.get(emotion, 0)
            
            # Higher score for matching emotions
            if text_score > 0.3 and face_score > 0.3:
                score += min(text_score, face_score) * 2
            else:
                # Penalty for mismatch
                score -= abs(text_score - face_score) * 0.5
        
        # Bonus for intensity match
        text_intensity = text_emotions.get('intensity', 0.5)
        face_intensity = face_emotions.get('intensity', 0.5)
        if abs(text_intensity - face_intensity) < 0.3:
            score += 0.5
        
        return max(0, score)
    
    def _create_emotion_layout(self, panels: List[Dict]) -> Dict:
        """Create layout with emotion-aware styling"""
        pages = []
        panels_per_page = 4
        
        for i in range(0, len(panels), panels_per_page):
            page_panels = panels[i:i+panels_per_page]
            
            page = {
                'width': 800,
                'height': 600,
                'panels': [],
                'bubbles': []
            }
            
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
                
                # Determine dominant emotion
                all_emotions = {**panel_data['text_emotions'], **panel_data['face_emotions']}
                dominant_emotion = max(all_emotions.items(), 
                                     key=lambda x: x[1] if x[0] != 'intensity' else 0)[0]
                
                # Add panel with emotion metadata
                page['panels'].append({
                    'x': x, 'y': y,
                    'width': w, 'height': h,
                    'image': panel_data['frame'],
                    'emotion': dominant_emotion,
                    'match_score': panel_data['match_score']
                })
                
                # Style bubble based on emotion
                bubble_style = self._get_emotion_bubble_style(dominant_emotion)
                
                page['bubbles'].append({
                    'id': f'bubble_{panel_data["index"]}',
                    'x': x + 20,
                    'y': y + h - 100,  # Position based on emotion
                    'width': 150,
                    'height': 70,
                    'text': panel_data['subtitle'].content,
                    'style': bubble_style
                })
            
            pages.append(page)
        
        return {'pages': pages}
    
    def _get_emotion_bubble_style(self, emotion: str) -> Dict:
        """Get bubble style for emotion"""
        styles = {
            'happy': {
                'shape': 'round',
                'border': '#4CAF50',
                'background': '#E8F5E9',
                'font': 'bold'
            },
            'sad': {
                'shape': 'droopy',
                'border': '#2196F3',
                'background': '#E3F2FD',
                'font': 'italic'
            },
            'angry': {
                'shape': 'jagged',
                'border': '#F44336',
                'background': '#FFEBEE',
                'font': 'bold',
                'size': 'large'
            },
            'surprised': {
                'shape': 'burst',
                'border': '#FF9800',
                'background': '#FFF3E0',
                'font': 'bold'
            },
            'neutral': {
                'shape': 'round',
                'border': '#333',
                'background': '#FFF',
                'font': 'normal'
            }
        }
        
        return styles.get(emotion, styles['neutral'])
    
    def _load_subtitles(self) -> List[srt.Subtitle]:
        """Load subtitles"""
        if os.path.exists('test1.srt'):
            with open('test1.srt', 'r') as f:
                return list(srt.parse(f.read()))
        return []
    
    def _get_all_frames(self) -> List[str]:
        """Get all available frames"""
        frames_dir = 'frames'
        if os.path.exists(frames_dir):
            frames = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir))
                     if f.endswith('.png')]
            return frames
        return []
    
    def _save_emotion_comic(self, comic_data: Dict):
        """Save emotion-aware comic"""
        os.makedirs('output', exist_ok=True)
        
        # Save JSON
        with open('output/emotion_comic.json', 'w') as f:
            json.dump(comic_data, f, indent=2)
        
        print("✅ Saved emotion-aware comic to output/emotion_comic.json")

# Test function
def create_emotion_comic(video_path='video/sample.mp4'):
    """Create an emotion-aware comic"""
    generator = EmotionAwareComicGenerator()
    return generator.generate_emotion_comic(video_path)

if __name__ == "__main__":
    create_emotion_comic()