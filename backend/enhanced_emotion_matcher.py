"""
Enhanced emotion matching with better text analysis and facial expression detection
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import re
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None
    print("⚠️ TextBlob not installed - using basic sentiment analysis")

class EnhancedEmotionMatcher:
    """Improved emotion matching between text and facial expressions"""
    
    def __init__(self):
        # Enhanced emotion keywords with weights
        self.emotion_keywords = {
            'happy': {
                'keywords': ['happy', 'joy', 'laugh', 'smile', 'fun', 'excited', 'yay', 'wow', 
                           'great', 'amazing', 'wonderful', 'love', 'beautiful', 'awesome'],
                'emojis': ['😊', '😄', '😃', '😁', '🙂', '😍', '❤️', '💕', '✨'],
                'punctuation': ['!', '!!', '!!!'],
                'weight': 1.0
            },
            'sad': {
                'keywords': ['sad', 'cry', 'tear', 'sorry', 'miss', 'lonely', 'hurt', 'pain',
                           'depressed', 'unhappy', 'disappointed', 'grief', 'sorrow'],
                'emojis': ['😢', '😭', '😔', '😞', '💔', '😿'],
                'punctuation': ['...'],
                'weight': 1.0
            },
            'angry': {
                'keywords': ['angry', 'mad', 'furious', 'hate', 'annoyed', 'frustrated',
                           'rage', 'irritated', 'stupid', 'damn', 'hell'],
                'emojis': ['😠', '😡', '🤬', '😤', '💢'],
                'punctuation': ['!?', '?!'],
                'weight': 1.0
            },
            'surprised': {
                'keywords': ['surprised', 'shock', 'what', 'oh', 'wow', 'really', 'seriously',
                           'unbelievable', 'impossible', 'amazing'],
                'emojis': ['😮', '😱', '😲', '🤯', '⚡'],
                'punctuation': ['?!', '!?', '???'],
                'weight': 0.9
            },
            'scared': {
                'keywords': ['scared', 'fear', 'afraid', 'terrified', 'frightened', 'horror',
                           'panic', 'worry', 'nervous', 'anxious'],
                'emojis': ['😨', '😰', '😱', '👻', '💀'],
                'punctuation': ['!!!', '...!'],
                'weight': 0.9
            },
            'neutral': {
                'keywords': ['okay', 'fine', 'alright', 'yes', 'no', 'maybe', 'sure'],
                'emojis': ['😐', '😑', '🙄'],
                'punctuation': ['.'],
                'weight': 0.5
            }
        }
        
        # Context modifiers
        self.intensifiers = ['very', 'so', 'really', 'extremely', 'super', 'totally']
        self.negations = ['not', 'no', "n't", 'never', 'neither', 'nor']
        
    def analyze_text_emotion(self, text: str) -> Dict[str, float]:
        """
        Enhanced text emotion analysis
        
        Returns emotions with confidence scores
        """
        text_lower = text.lower()
        emotions = {emotion: 0.0 for emotion in self.emotion_keywords}
        
        # 1. Keyword matching with context
        for emotion, data in self.emotion_keywords.items():
            score = 0.0
            
            # Check keywords
            for keyword in data['keywords']:
                if keyword in text_lower:
                    # Check for negation
                    if self._is_negated(text_lower, keyword):
                        # Negated emotion might indicate opposite
                        opposite = self._get_opposite_emotion(emotion)
                        if opposite:
                            emotions[opposite] += 0.3
                    else:
                        score += 0.5
                        
                        # Check for intensifiers
                        if self._has_intensifier(text_lower, keyword):
                            score += 0.3
            
            # Check punctuation patterns
            for punct in data['punctuation']:
                if punct in text:
                    score += 0.2
            
            # Weight the score
            emotions[emotion] = min(score * data['weight'], 1.0)
        
        # 2. Sentiment analysis using TextBlob (if available)
        intensity = 0.5
        if TextBlob:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to 1
                subjectivity = blob.sentiment.subjectivity  # 0 to 1
                
                # Map polarity to emotions
                if polarity > 0.3:
                    emotions['happy'] += polarity * 0.5
                elif polarity < -0.3:
                    emotions['sad'] += abs(polarity) * 0.3
                    emotions['angry'] += abs(polarity) * 0.2
                
                # High subjectivity might indicate stronger emotion
                intensity = subjectivity * 0.5
                
            except:
                intensity = 0.5
        else:
            # Simple polarity based on keywords if TextBlob not available
            positive_words = ['good', 'great', 'love', 'happy', 'wonderful', 'amazing']
            negative_words = ['bad', 'hate', 'sad', 'angry', 'terrible', 'awful']
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                emotions['happy'] += 0.3
            elif neg_count > pos_count:
                emotions['sad'] += 0.2
                emotions['angry'] += 0.1
        
        # 3. Exclamation marks indicate intensity
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            intensity = min(1.0, 0.5 + exclamation_count * 0.2)
        
        # 4. Question marks might indicate surprise or confusion
        if '?' in text:
            emotions['surprised'] += 0.3
        
        # 5. Normalize scores
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        else:
            emotions['neutral'] = 1.0
        
        # Add intensity
        emotions['intensity'] = intensity
        
        return emotions
    
    def _is_negated(self, text: str, keyword: str) -> bool:
        """Check if keyword is negated"""
        # Simple check - look for negation words before keyword
        keyword_pos = text.find(keyword)
        if keyword_pos > 0:
            before_text = text[:keyword_pos].split()[-3:]  # Last 3 words before keyword
            return any(neg in before_text for neg in self.negations)
        return False
    
    def _has_intensifier(self, text: str, keyword: str) -> bool:
        """Check if keyword has intensifier"""
        keyword_pos = text.find(keyword)
        if keyword_pos > 0:
            before_text = text[:keyword_pos].split()[-2:]  # Last 2 words before keyword
            return any(intensifier in before_text for intensifier in self.intensifiers)
        return False
    
    def _get_opposite_emotion(self, emotion: str) -> str:
        """Get opposite emotion"""
        opposites = {
            'happy': 'sad',
            'sad': 'happy',
            'angry': 'happy',
            'scared': 'confident',
            'confident': 'scared'
        }
        return opposites.get(emotion, 'neutral')
    
    def match_frames_to_emotions(self, frames: List[str], subtitles: List, 
                                eye_detector=None) -> List[Dict]:
        """
        Match frames to subtitles based on emotions and eye state
        
        Returns list of matched panels with metadata
        """
        from backend.emotion_aware_comic import FacialExpressionAnalyzer
        face_analyzer = FacialExpressionAnalyzer()
        
        matched_panels = []
        
        for sub in subtitles:
            # Analyze text emotion
            text_emotions = self.analyze_text_emotion(sub.content)
            
            # Find time range for frames
            start_frame = int(sub.index * len(frames) / len(subtitles))
            end_frame = min(start_frame + 5, len(frames))  # Check up to 5 frames
            
            best_match = None
            best_score = -1
            
            for i in range(start_frame, end_frame):
                if i >= len(frames):
                    break
                
                frame_path = frames[i]
                
                # Check eye state if detector available
                eye_score = 1.0
                if eye_detector:
                    eye_state = eye_detector.check_eyes_state(frame_path)
                    if eye_state['state'] == 'open':
                        eye_score = 1.2  # Bonus for open eyes
                    elif eye_state['state'] == 'half_closed':
                        eye_score = 0.5  # Penalty for half-closed
                    elif eye_state['state'] == 'closed':
                        eye_score = 0.1  # Strong penalty for closed
                
                # Analyze facial expression
                face_emotions = face_analyzer.analyze_expression(frame_path)
                
                # Calculate match score
                emotion_score = self._calculate_match_score(text_emotions, face_emotions)
                total_score = emotion_score * eye_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_match = {
                        'frame': frame_path,
                        'subtitle': sub,
                        'text_emotions': text_emotions,
                        'face_emotions': face_emotions,
                        'match_score': total_score,
                        'eye_score': eye_score
                    }
            
            if best_match:
                matched_panels.append(best_match)
                print(f"  ✅ Matched: '{sub.content[:30]}...' - Score: {best_score:.2f}")
        
        return matched_panels
    
    def _calculate_match_score(self, text_emotions: Dict, face_emotions: Dict) -> float:
        """Calculate emotion match score with improved algorithm"""
        score = 0.0
        
        # Get top emotions from each
        text_top = sorted([(k, v) for k, v in text_emotions.items() if k != 'intensity'], 
                         key=lambda x: x[1], reverse=True)[:2]
        face_top = sorted([(k, v) for k, v in face_emotions.items() if k != 'intensity'], 
                         key=lambda x: x[1], reverse=True)[:2]
        
        # Check if top emotions match
        if text_top and face_top:
            # Exact match of top emotion
            if text_top[0][0] == face_top[0][0]:
                score += 1.0 * min(text_top[0][1], face_top[0][1])
            
            # Check secondary emotions
            for t_emotion, t_score in text_top:
                for f_emotion, f_score in face_top:
                    if t_emotion == f_emotion:
                        score += 0.5 * min(t_score, f_score)
        
        # Penalty for conflicting emotions
        if text_emotions.get('happy', 0) > 0.5 and face_emotions.get('sad', 0) > 0.5:
            score -= 0.5
        if text_emotions.get('sad', 0) > 0.5 and face_emotions.get('happy', 0) > 0.5:
            score -= 0.5
        
        # Intensity matching
        text_intensity = text_emotions.get('intensity', 0.5)
        face_intensity = face_emotions.get('intensity', 0.5)
        intensity_diff = abs(text_intensity - face_intensity)
        score += (1 - intensity_diff) * 0.3
        
        return max(0, score)