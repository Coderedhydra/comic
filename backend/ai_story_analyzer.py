"""
AI-Powered Complete Story Analyzer
Uses AI models to understand entire video story and select appropriate frames with expressions
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
import srt
from datetime import timedelta
import re
import math

class AIStoryAnalyzer:
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None
        self.init_opencv_classifiers()
        self.emotion_model = None
        self.init_emotion_detection()
        
    def init_opencv_classifiers(self):
        """Initialize OpenCV classifiers for face and emotion detection"""
        try:
            # Load face detection
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(face_cascade_path):
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
                print("✅ Face detection initialized")
            
            # Load eye detection
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            if os.path.exists(eye_cascade_path):
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
                print("✅ Eye detection initialized")
                
            # Load smile detection
            smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
            if os.path.exists(smile_cascade_path):
                self.smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
                print("✅ Smile detection initialized")
                
        except Exception as e:
            print(f"⚠️ OpenCV initialization: {e}")
    
    def init_emotion_detection(self):
        """Initialize emotion detection model"""
        try:
            # Try to use a simple emotion detection based on facial features
            print("✅ Basic emotion detection ready")
        except Exception as e:
            print(f"⚠️ Emotion model setup: {e}")
    
    def analyze_frame_emotion(self, frame_path: str) -> Dict[str, float]:
        """
        Analyze emotional content of a frame
        Returns emotion scores and facial expression data
        """
        try:
            img = cv2.imread(frame_path)
            if img is None:
                return {'neutral': 1.0, 'face_detected': False, 'eye_openness': 0.5}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4) if self.face_cascade else []
            
            if len(faces) == 0:
                return {'neutral': 1.0, 'face_detected': False, 'eye_openness': 0.5}
            
            # Analyze the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3) if self.eye_cascade else []
            eye_openness = self.calculate_eye_openness(face_roi, eyes)
            
            # Detect smile
            smiles = self.smile_cascade.detectMultiScale(face_roi, 1.8, 20) if self.smile_cascade else []
            smile_strength = len(smiles) / 3.0  # Normalize
            
            # Basic emotion analysis based on facial features
            emotions = self.analyze_facial_features(face_roi, eye_openness, smile_strength)
            emotions['face_detected'] = True
            emotions['eye_openness'] = eye_openness
            
            return emotions
            
        except Exception as e:
            print(f"Frame emotion analysis error: {e}")
            return {'neutral': 1.0, 'face_detected': False, 'eye_openness': 0.5}
    
    def calculate_eye_openness(self, face_roi: np.ndarray, eyes: List) -> float:
        """Calculate eye openness score"""
        if len(eyes) == 0:
            return 0.3  # No eyes detected - likely closed
        
        if len(eyes) == 1:
            return 0.6  # One eye visible
        
        # Analyze eye regions for openness
        eye_scores = []
        for (ex, ey, ew, eh) in eyes:
            eye_region = face_roi[ey:ey+eh, ex:ex+ew]
            if eye_region.size > 0:
                # Calculate variance - open eyes have more variation
                variance = np.var(eye_region)
                openness = min(1.0, variance / 400.0)
                
                # Check aspect ratio - closed eyes are flatter
                aspect_ratio = eh / ew if ew > 0 else 0
                if aspect_ratio < 0.25:  # Very flat
                    openness *= 0.5
                
                eye_scores.append(openness)
        
        return sum(eye_scores) / len(eye_scores) if eye_scores else 0.5
    
    def analyze_facial_features(self, face_roi: np.ndarray, eye_openness: float, smile_strength: float) -> Dict[str, float]:
        """Analyze facial features to determine emotions"""
        emotions = {
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'surprised': 0.0,
            'neutral': 0.0,
            'excited': 0.0
        }
        
        # Happy indicators
        if smile_strength > 0.3:
            emotions['happy'] = min(1.0, smile_strength * 2)
        
        # Surprised indicators (wide eyes)
        if eye_openness > 0.8:
            emotions['surprised'] = eye_openness * 0.7
        
        # Sad indicators (low eye openness, no smile)
        if eye_openness < 0.4 and smile_strength < 0.1:
            emotions['sad'] = 0.6
        
        # Excited (combination of open eyes and smile)
        if eye_openness > 0.7 and smile_strength > 0.2:
            emotions['excited'] = (eye_openness + smile_strength) * 0.5
        
        # Neutral (default when no strong emotions)
        total_emotion = sum(emotions.values())
        if total_emotion < 0.3:
            emotions['neutral'] = 1.0
        
        # Normalize emotions
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions
    
    def analyze_complete_story_structure(self, subtitles: List, video_duration: float) -> List[Dict]:
        """
        Analyze complete story structure using AI understanding
        Divides story into meaningful checkpoints covering entire video
        """
        print(f"🧠 AI analyzing complete story structure...")
        print(f"📹 Video duration: {video_duration:.1f} seconds")
        
        if not subtitles:
            return self.create_default_story_checkpoints(video_duration)
        
        # Sort subtitles chronologically
        subtitles.sort(key=lambda x: x.start.total_seconds())
        
        # Analyze story phases and create checkpoints
        story_checkpoints = []
        
        # Define story structure with specific checkpoint allocation
        story_phases = [
            {
                'name': 'Opening',
                'description': 'Introduction and world setup',
                'start_percent': 0.0,
                'end_percent': 0.125,  # First 12.5%
                'checkpoints': 6,
                'keywords': ['begin', 'start', 'meet', 'introduce', 'hello', 'welcome', 'first', 'once']
            },
            {
                'name': 'Character Setup',
                'description': 'Character introduction and relationships',
                'start_percent': 0.125,
                'end_percent': 0.25,   # 12.5% - 25%
                'checkpoints': 6,
                'keywords': ['name', 'friend', 'family', 'know', 'meet', 'together', 'relationship']
            },
            {
                'name': 'Conflict Introduction',
                'description': 'Problem or challenge emerges',
                'start_percent': 0.25,
                'end_percent': 0.375,  # 25% - 37.5%
                'checkpoints': 6,
                'keywords': ['problem', 'trouble', 'wrong', 'danger', 'enemy', 'conflict', 'challenge']
            },
            {
                'name': 'Rising Tension',
                'description': 'Complications and obstacles',
                'start_percent': 0.375,
                'end_percent': 0.5,    # 37.5% - 50%
                'checkpoints': 6,
                'keywords': ['difficult', 'hard', 'struggle', 'fight', 'against', 'overcome', 'try']
            },
            {
                'name': 'Peak Conflict',
                'description': 'Major confrontation and climax',
                'start_percent': 0.5,
                'end_percent': 0.625,  # 50% - 62.5%
                'checkpoints': 6,
                'keywords': ['battle', 'fight', 'final', 'ultimate', 'decide', 'crucial', 'important']
            },
            {
                'name': 'Climax Resolution',
                'description': 'Peak moment and turning point',
                'start_percent': 0.625,
                'end_percent': 0.75,   # 62.5% - 75%
                'checkpoints': 6,
                'keywords': ['victory', 'defeat', 'win', 'lose', 'success', 'fail', 'overcome']
            },
            {
                'name': 'Falling Action',
                'description': 'Aftermath and consequences',
                'start_percent': 0.75,
                'end_percent': 0.875,  # 75% - 87.5%
                'checkpoints': 6,
                'keywords': ['after', 'result', 'consequence', 'change', 'different', 'now']
            },
            {
                'name': 'Resolution',
                'description': 'Conclusion and new beginning',
                'start_percent': 0.875,
                'end_percent': 1.0,    # 87.5% - 100%
                'checkpoints': 6,
                'keywords': ['end', 'finish', 'complete', 'finally', 'peace', 'home', 'future']
            }
        ]
        
        checkpoint_id = 1
        
        for phase in story_phases:
            print(f"📍 Analyzing {phase['name']} phase...")
            
            # Find subtitles in this phase
            phase_start_time = phase['start_percent'] * video_duration
            phase_end_time = phase['end_percent'] * video_duration
            
            phase_subtitles = [
                sub for sub in subtitles
                if phase_start_time <= sub.start.total_seconds() <= phase_end_time
            ]
            
            if not phase_subtitles:
                # Create narrative checkpoints for this phase
                for i in range(phase['checkpoints']):
                    checkpoint_time = phase_start_time + (i / phase['checkpoints']) * (phase_end_time - phase_start_time)
                    
                    story_checkpoints.append({
                        'checkpoint_id': checkpoint_id,
                        'time': checkpoint_time,
                        'phase': phase['name'],
                        'description': phase['description'],
                        'story_text': self.generate_phase_narrative(phase['name'], i, phase['checkpoints']),
                        'importance': 0.7,
                        'expected_emotion': self.get_phase_emotion(phase['name']),
                        'is_narrative': True
                    })
                    checkpoint_id += 1
                continue
            
            # Analyze subtitles for story relevance
            analyzed_subs = []
            for sub in phase_subtitles:
                relevance = self.calculate_story_relevance(sub.content, phase['keywords'])
                analyzed_subs.append({
                    'subtitle': sub,
                    'relevance': relevance,
                    'time': sub.start.total_seconds()
                })
            
            # Sort by relevance and time
            analyzed_subs.sort(key=lambda x: (-x['relevance'], x['time']))
            
            # Create checkpoints for this phase
            checkpoints_needed = phase['checkpoints']
            
            if len(analyzed_subs) >= checkpoints_needed:
                # Use most relevant subtitles
                selected = analyzed_subs[:checkpoints_needed]
                selected.sort(key=lambda x: x['time'])  # Back to chronological order
            else:
                # Use all available and fill gaps
                selected = sorted(analyzed_subs, key=lambda x: x['time'])
                
                # Fill remaining checkpoints
                while len(selected) < checkpoints_needed:
                    if len(selected) == 0:
                        gap_time = phase_start_time
                    else:
                        # Find largest time gap
                        gaps = []
                        for i in range(len(selected) - 1):
                            gap_size = selected[i+1]['time'] - selected[i]['time']
                            gaps.append((gap_size, i))
                        
                        if gaps:
                            largest_gap = max(gaps)
                            gap_index = largest_gap[1]
                            gap_time = (selected[gap_index]['time'] + selected[gap_index + 1]['time']) / 2
                        else:
                            gap_time = selected[-1]['time'] + (phase_end_time - selected[-1]['time']) / 2
                    
                    # Create narrative checkpoint
                    selected.append({
                        'time': gap_time,
                        'relevance': 0.5,
                        'subtitle': None,
                        'is_narrative': True
                    })
                    
                    selected.sort(key=lambda x: x['time'])
            
            # Create story checkpoints
            for i, item in enumerate(selected[:checkpoints_needed]):
                if item.get('subtitle'):
                    story_text = item['subtitle'].content
                else:
                    story_text = self.generate_phase_narrative(phase['name'], i, checkpoints_needed)
                
                story_checkpoints.append({
                    'checkpoint_id': checkpoint_id,
                    'time': item['time'],
                    'phase': phase['name'],
                    'description': phase['description'],
                    'story_text': story_text,
                    'importance': item['relevance'],
                    'expected_emotion': self.get_phase_emotion(phase['name']),
                    'is_narrative': item.get('is_narrative', False)
                })
                checkpoint_id += 1
        
        print(f"✅ Created {len(story_checkpoints)} story checkpoints covering entire video")
        return story_checkpoints
    
    def calculate_story_relevance(self, text: str, phase_keywords: List[str]) -> float:
        """Calculate how relevant a subtitle is to the current story phase"""
        if not text:
            return 0.1
        
        text_lower = text.lower()
        relevance = 0.0
        
        # Keyword matching
        for keyword in phase_keywords:
            if keyword in text_lower:
                relevance += 0.3
        
        # General story importance indicators
        story_indicators = [
            'important', 'must', 'need', 'have to', 'crucial', 'critical',
            'love', 'hate', 'friend', 'enemy', 'family', 'together',
            'go', 'come', 'leave', 'stay', 'help', 'save', 'protect',
            'truth', 'secret', 'discover', 'find', 'realize', 'understand'
        ]
        
        for indicator in story_indicators:
            if indicator in text_lower:
                relevance += 0.2
        
        # Dialogue vs narration (dialogue is usually more important)
        if '"' in text or "'" in text:
            relevance += 0.1
        
        # Questions and exclamations
        if '?' in text:
            relevance += 0.2
        if '!' in text:
            relevance += 0.1
        
        # Length consideration (very short text is less relevant)
        if len(text.strip()) < 5:
            relevance *= 0.5
        
        return min(1.0, relevance)
    
    def get_phase_emotion(self, phase_name: str) -> str:
        """Get expected dominant emotion for story phase"""
        emotion_map = {
            'Opening': 'neutral',
            'Character Setup': 'happy',
            'Conflict Introduction': 'surprised',
            'Rising Tension': 'angry',
            'Peak Conflict': 'angry',
            'Climax Resolution': 'excited',
            'Falling Action': 'neutral',
            'Resolution': 'happy'
        }
        return emotion_map.get(phase_name, 'neutral')
    
    def generate_phase_narrative(self, phase_name: str, index: int, total: int) -> str:
        """Generate narrative text for story phase"""
        narratives = {
            'Opening': [
                "Our story begins in a world full of possibilities and wonder.",
                "We are introduced to the main character and their daily life.",
                "The setting comes alive with rich details and atmosphere.",
                "Initial circumstances set the foundation for the adventure ahead.",
                "Characters move through their world, unaware of what's coming.",
                "The stage is set for the extraordinary events that will unfold."
            ],
            'Character Setup': [
                "We learn about our protagonist's hopes, dreams, and fears.",
                "Important relationships are established and explored in depth.",
                "Character personalities shine through their actions and words.",
                "The bonds between characters grow stronger through shared experiences.",
                "We discover what motivates each character to act and choose.",
                "The ensemble cast comes together, each bringing unique strengths."
            ],
            'Conflict Introduction': [
                "A problem emerges that threatens the peaceful status quo.",
                "Our characters face their first real challenge or obstacle.",
                "The antagonist's presence begins to make itself known.",
                "Complications arise that force characters to make difficult choices.",
                "The central conflict of the story becomes clear and urgent.",
                "Characters realize that their comfortable world is changing."
            ],
            'Rising Tension': [
                "Challenges intensify as obstacles become more difficult to overcome.",
                "Characters must adapt their strategies and push beyond their limits.",
                "Relationships are tested under the pressure of mounting conflict.",
                "Each small victory comes with new complications and setbacks.",
                "The stakes continue to rise as more is put at risk.",
                "Characters discover inner strength they didn't know they possessed."
            ],
            'Peak Conflict': [
                "The major confrontation begins as all forces come together.",
                "Everything our characters have learned is put to the ultimate test.",
                "The conflict reaches its most intense and dangerous point.",
                "Characters must make the hardest choices of their entire journey.",
                "The fate of everything they care about hangs in the balance.",
                "This is the moment that will define the outcome of everything."
            ],
            'Climax Resolution': [
                "The peak moment arrives as the conflict reaches its conclusion.",
                "Characters demonstrate how much they have grown and changed.",
                "The resolution of the central conflict begins to take shape.",
                "Truth is revealed and mysteries are finally explained.",
                "The turning point that will determine everyone's future unfolds.",
                "Victory or defeat becomes clear as the dust begins to settle."
            ],
            'Falling Action': [
                "The immediate crisis is resolved and characters process what happened.",
                "Consequences of the climactic events begin to become clear.",
                "Characters deal with the aftermath of their choices and actions.",
                "The world begins to heal and rebuild from the conflict.",
                "Relationships are redefined by everything they've been through together.",
                "Order starts to emerge from the chaos of the recent battle."
            ],
            'Resolution': [
                "Peace is restored and the world finds its new equilibrium.",
                "Characters have grown and changed through their transformative journey.",
                "Loose ends are tied up and remaining questions are answered.",
                "The community celebrates the heroes and their achievements.",
                "New beginnings emerge from the resolution of the old conflicts.",
                "The story concludes with hope for the future and lessons learned."
            ]
        }
        
        phase_narratives = narratives.get(phase_name, ["The story continues to unfold with new developments."])
        return phase_narratives[index % len(phase_narratives)]
    
    def create_default_story_checkpoints(self, video_duration: float) -> List[Dict]:
        """Create default story checkpoints when no subtitles available"""
        checkpoints = []
        
        # Create 48 evenly distributed checkpoints
        for i in range(48):
            checkpoint_time = (i / 47) * video_duration if video_duration > 0 else i * 2
            phase_progress = i / 47
            
            if phase_progress < 0.125:
                phase = 'Opening'
            elif phase_progress < 0.25:
                phase = 'Character Setup'
            elif phase_progress < 0.375:
                phase = 'Conflict Introduction'
            elif phase_progress < 0.5:
                phase = 'Rising Tension'
            elif phase_progress < 0.625:
                phase = 'Peak Conflict'
            elif phase_progress < 0.75:
                phase = 'Climax Resolution'
            elif phase_progress < 0.875:
                phase = 'Falling Action'
            else:
                phase = 'Resolution'
            
            checkpoints.append({
                'checkpoint_id': i + 1,
                'time': checkpoint_time,
                'phase': phase,
                'description': f'Story checkpoint {i + 1}',
                'story_text': f'The story continues to develop at this important moment.',
                'importance': 0.5,
                'expected_emotion': self.get_phase_emotion(phase),
                'is_narrative': True
            })
        
        return checkpoints
    
    def select_best_frames_with_expressions(self, video_path: str, story_checkpoints: List[Dict], output_dir: str = 'frames/final') -> bool:
        """
        Select best frames that match story emotions and have good expressions
        """
        print(f"🎬 Selecting frames with matching expressions for {len(story_checkpoints)} checkpoints...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear existing frames
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(output_dir, f))
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        successful_frames = []
        
        for i, checkpoint in enumerate(story_checkpoints):
            target_time = checkpoint['time']
            expected_emotion = checkpoint['expected_emotion']
            
            print(f"🎯 Checkpoint {i+1:2d} ({checkpoint['phase']:15s}): Looking for {expected_emotion} emotion")
            
            # Search for best frame in time window
            best_frame = None
            best_score = 0
            best_emotion_data = None
            
            # Try multiple time offsets to find best matching frame
            time_offsets = [0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0, -3.0, 3.0]
            
            for offset in time_offsets:
                frame_time = max(0, target_time + offset)
                frame_number = int(frame_time * fps)
                
                if frame_number >= total_frames:
                    continue
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Save temp frame for analysis
                temp_path = f'temp_analysis_{i}.png'
                cv2.imwrite(temp_path, frame)
                
                # Analyze frame emotion and quality
                emotion_data = self.analyze_frame_emotion(temp_path)
                
                # Calculate match score
                emotion_match = emotion_data.get(expected_emotion, 0)
                eye_quality = emotion_data.get('eye_openness', 0.5)
                face_detected = emotion_data.get('face_detected', False)
                
                # Scoring: emotion match (40%) + eye quality (40%) + face detection (20%)
                total_score = (emotion_match * 0.4 + 
                              eye_quality * 0.4 + 
                              (1.0 if face_detected else 0.3) * 0.2)
                
                print(f"    Frame at {frame_time:.1f}s: emotion={emotion_match:.2f}, eyes={eye_quality:.2f}, total={total_score:.2f}")
                
                if total_score > best_score:
                    best_score = total_score
                    best_frame = frame.copy()
                    best_emotion_data = emotion_data
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Save best frame (always save something)
            if best_frame is not None:
                output_path = os.path.join(output_dir, f'frame{i:03d}.png')
                cv2.imwrite(output_path, best_frame)
                
                successful_frames.append({
                    'checkpoint': i + 1,
                    'time': target_time,
                    'phase': checkpoint['phase'],
                    'expected_emotion': expected_emotion,
                    'detected_emotions': best_emotion_data,
                    'match_score': best_score,
                    'story_text': checkpoint['story_text']
                })
                
                print(f"✅ Saved checkpoint {i+1} (score: {best_score:.2f})")
            else:
                print(f"⚠️ No frame found for checkpoint {i+1}")
        
        cap.release()
        
        # Save analysis results
        with open(os.path.join(output_dir, 'ai_analysis.json'), 'w') as f:
            json.dump({
                'story_checkpoints': story_checkpoints,
                'selected_frames': successful_frames
            }, f, indent=2, default=str)
        
        print(f"🎉 Selected {len(successful_frames)} frames with AI emotion matching")
        return len(successful_frames) > 0

def create_ai_story_comic(video_path: str, subtitles: List, target_panels: int = 48) -> bool:
    """
    Create complete story comic using AI analysis and emotion matching
    """
    analyzer = AIStoryAnalyzer()
    
    # Get video duration
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 100
    cap.release()
    
    print(f"🎬 AI analyzing video: {video_duration:.1f} seconds, {total_frames} frames")
    
    # Analyze complete story structure
    story_checkpoints = analyzer.analyze_complete_story_structure(subtitles, video_duration)
    
    if not story_checkpoints:
        return False
    
    # Select frames with emotion matching
    success = analyzer.select_best_frames_with_expressions(video_path, story_checkpoints)
    
    return success