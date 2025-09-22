"""
Emotion-based keyframe selection with eye state detection
Selects frames based on emotional content and avoids closed/half-closed eyes
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict
import json

class EmotionKeyframeSelector:
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.init_opencv_classifiers()
    
    def init_opencv_classifiers(self):
        """Initialize OpenCV face and eye detection classifiers"""
        try:
            # Try to load face and eye cascades
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            
            if os.path.exists(face_cascade_path):
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
                print("✅ Face detection loaded")
            
            if os.path.exists(eye_cascade_path):
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
                print("✅ Eye detection loaded")
                
        except Exception as e:
            print(f"⚠️ OpenCV classifiers not available: {e}")
    
    def detect_eye_state(self, image_path: str) -> float:
        """
        Detect eye state in image
        Returns: 0.0-1.0 score (1.0 = eyes wide open, 0.0 = eyes closed)
        """
        try:
            if not self.face_cascade or not self.eye_cascade:
                return 0.8  # Default assume good eyes if detection not available
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return 0.8
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return 0.7  # No face detected, assume okay
            
            best_eye_score = 0.0
            
            for (x, y, w, h) in faces:
                # Region of interest for eyes (upper half of face)
                roi_gray = gray[y:y+int(h*0.6), x:x+w]
                roi_color = img[y:y+int(h*0.6), x:x+w]
                
                # Detect eyes in face region
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                if len(eyes) >= 2:
                    # Good - detected both eyes
                    eye_score = 1.0
                    
                    # Check eye openness by analyzing the eye regions
                    for (ex, ey, ew, eh) in eyes:
                        eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                        
                        # Simple openness check based on variance
                        # Open eyes have more variation in pixel values
                        if eye_region.size > 0:
                            variance = np.var(eye_region)
                            # Normalize variance (higher variance = more open)
                            openness = min(1.0, variance / 1000.0)
                            eye_score = min(eye_score, openness)
                    
                    best_eye_score = max(best_eye_score, eye_score)
                
                elif len(eyes) == 1:
                    # Only one eye detected - might be profile or partially closed
                    best_eye_score = max(best_eye_score, 0.6)
                else:
                    # No eyes detected in face - likely closed or very small
                    best_eye_score = max(best_eye_score, 0.3)
            
            return best_eye_score
            
        except Exception as e:
            print(f"Eye detection error for {image_path}: {e}")
            return 0.5  # Default middle score
    
    def analyze_emotion_from_text(self, text: str) -> Dict[str, float]:
        """
        Analyze emotional content of text
        Returns emotion scores for different emotions
        """
        if not text:
            return {'neutral': 1.0}
        
        text = text.lower()
        
        # Emotion keywords
        emotion_keywords = {
            'happy': ['happy', 'joy', 'smile', 'laugh', 'excited', 'wonderful', 'great', 'amazing', 'love', 'celebration'],
            'sad': ['sad', 'cry', 'tears', 'sorrow', 'grief', 'depressed', 'lonely', 'hurt', 'pain', 'loss'],
            'angry': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'frustrated', 'fight', 'battle', 'war'],
            'surprised': ['surprised', 'shock', 'amazed', 'wow', 'incredible', 'unbelievable', 'sudden', 'unexpected'],
            'scared': ['scared', 'afraid', 'fear', 'terror', 'frightened', 'worried', 'anxious', 'panic', 'danger'],
            'neutral': ['said', 'told', 'spoke', 'mentioned', 'explained', 'described', 'stated']
        }
        
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = 0
            for keyword in keywords:
                score += text.count(keyword)
            emotion_scores[emotion] = score
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total
        else:
            emotion_scores['neutral'] = 1.0
        
        return emotion_scores
    
    def select_emotion_keyframes(self, video_path: str, subtitles: List, output_dir: str = 'frames/final', max_frames: int = 48) -> bool:
        """
        Select keyframes based on emotional content and eye state
        """
        try:
            print(f"🎭 Starting emotion-based keyframe selection...")
            print(f"📝 Processing {len(subtitles)} subtitle entries")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Clear existing frames
            for f in os.listdir(output_dir):
                if f.endswith('.png'):
                    os.remove(os.path.join(output_dir, f))
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📹 Video: {fps} FPS, {total_frames} frames")
            
            selected_frames = []
            
            for i, subtitle in enumerate(subtitles[:max_frames]):
                try:
                    # Convert subtitle time to frame number
                    start_time = subtitle.start.total_seconds()
                    end_time = subtitle.end.total_seconds()
                    mid_time = (start_time + end_time) / 2
                    
                    # Get emotion analysis
                    emotion_scores = self.analyze_emotion_from_text(subtitle.content)
                    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                    
                    print(f"📝 Subtitle {i+1}: '{subtitle.content[:50]}...' -> {dominant_emotion}")
                    
                    # Try multiple frames around the subtitle time
                    best_frame = None
                    best_score = 0
                    
                    # Check frames in a window around the subtitle
                    for time_offset in [0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5]:
                        frame_time = mid_time + time_offset
                        if frame_time < 0:
                            continue
                        
                        frame_number = int(frame_time * fps)
                        if frame_number >= total_frames:
                            continue
                        
                        # Extract frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = cap.read()
                        
                        if not ret:
                            continue
                        
                        # Save temporary frame for analysis
                        temp_path = f'temp_frame_{i}.png'
                        cv2.imwrite(temp_path, frame)
                        
                        # Analyze eye state
                        eye_score = self.detect_eye_state(temp_path)
                        
                        # Calculate total score (eye state is most important)
                        total_score = eye_score * 0.8 + emotion_scores.get(dominant_emotion, 0) * 0.2
                        
                        print(f"  Frame {frame_number}: eye_score={eye_score:.2f}, total_score={total_score:.2f}")
                        
                        if total_score > best_score and eye_score > 0.6:  # Minimum eye threshold
                            best_score = total_score
                            best_frame = frame.copy()
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    
                    # Save best frame
                    if best_frame is not None:
                        output_path = os.path.join(output_dir, f'frame{i:03d}.png')
                        cv2.imwrite(output_path, best_frame)
                        selected_frames.append({
                            'frame': i,
                            'emotion': dominant_emotion,
                            'eye_score': best_score,
                            'text': subtitle.content
                        })
                        print(f"✅ Saved frame {i:03d} (emotion: {dominant_emotion}, score: {best_score:.2f})")
                    else:
                        print(f"⚠️ No suitable frame found for subtitle {i+1}")
                
                except Exception as e:
                    print(f"❌ Error processing subtitle {i}: {e}")
                    continue
            
            cap.release()
            
            print(f"🎉 Selected {len(selected_frames)} emotion-based keyframes with good eye states")
            
            # Save selection metadata
            with open(os.path.join(output_dir, 'emotion_selection.json'), 'w') as f:
                json.dump(selected_frames, f, indent=2)
            
            return len(selected_frames) > 0
            
        except Exception as e:
            print(f"❌ Emotion keyframe selection failed: {e}")
            return False

def generate_emotion_keyframes(video_path: str, subtitles: List, max_frames: int = 48) -> bool:
    """
    Main function to generate emotion-based keyframes with eye detection
    """
    selector = EmotionKeyframeSelector()
    return selector.select_emotion_keyframes(video_path, subtitles, max_frames=max_frames)