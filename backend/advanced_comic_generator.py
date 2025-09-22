"""
Advanced Comic Generator based on automated-comic-generation-with-ai-enhancement pattern
Implements comprehensive video analysis, AI enhancement, and editable bubble system
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

class AdvancedComicGenerator:
    def __init__(self):
        self.video_path = None
        self.output_dir = 'frames/final'
        self.comic_data = {
            'pages': [],
            'metadata': {},
            'ai_analysis': {},
            'editable_elements': []
        }
        
        # Initialize AI components
        self.face_detector = None
        self.emotion_analyzer = None
        self.story_analyzer = None
        self.init_ai_components()
        
    def init_ai_components(self):
        """Initialize AI components for enhancement"""
        try:
            # Face detection setup
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(face_cascade_path):
                self.face_detector = cv2.CascadeClassifier(face_cascade_path)
                print("✅ Face detector initialized")
            
            # Emotion analysis setup (using OpenCV + custom logic)
            self.emotion_analyzer = EmotionAnalyzer()
            print("✅ Emotion analyzer initialized")
            
            # Story analysis setup
            self.story_analyzer = StoryAnalyzer()
            print("✅ Story analyzer initialized")
            
        except Exception as e:
            print(f"⚠️ AI component initialization: {e}")
    
    def process_video_to_comic(self, video_path: str, subtitles: List = None, target_panels: int = 48) -> Dict:
        """
        Main processing pipeline: Video → AI Analysis → Comic Generation
        """
        self.video_path = video_path
        
        print("🎬 Starting Advanced Comic Generation Pipeline...")
        print("=" * 60)
        
        # Step 1: Video Analysis and Preprocessing
        video_info = self.analyze_video(video_path)
        print(f"📹 Video: {video_info['duration']:.1f}s, {video_info['fps']:.1f} FPS, {video_info['total_frames']} frames")
        
        # Step 2: Story Structure Analysis
        story_structure = self.story_analyzer.analyze_complete_story(subtitles, video_info['duration'])
        print(f"📖 Story structure: {len(story_structure['phases'])} phases, {len(story_structure['checkpoints'])} checkpoints")
        
        # Step 3: AI-Enhanced Frame Selection
        selected_frames = self.select_frames_with_ai_enhancement(video_path, story_structure, target_panels)
        print(f"🎯 Selected {len(selected_frames)} frames with AI enhancement")
        
        # Step 4: Generate Editable Comic Structure
        comic_structure = self.generate_editable_comic_structure(selected_frames, story_structure)
        print(f"📚 Generated comic: {len(comic_structure['pages'])} pages, {sum(len(p['panels']) for p in comic_structure['pages'])} panels")
        
        # Step 5: Create Enhanced Bubbles
        enhanced_bubbles = self.create_ai_enhanced_bubbles(selected_frames, story_structure)
        print(f"💬 Created {len(enhanced_bubbles)} AI-enhanced bubbles")
        
        # Step 6: Assemble Final Comic
        final_comic = self.assemble_final_comic(comic_structure, enhanced_bubbles)
        
        # Step 7: Save All Data
        self.save_comic_data(final_comic)
        
        print("✅ Advanced Comic Generation Complete!")
        return final_comic
    
    def analyze_video(self, video_path: str) -> Dict:
        """Analyze video properties and characteristics"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Sample frames for analysis
        frame_samples = []
        sample_count = min(20, total_frames // 100)  # Sample every 5% or 20 frames max
        
        for i in range(sample_count):
            frame_pos = int((i / (sample_count - 1)) * total_frames) if sample_count > 1 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                frame_samples.append({
                    'position': frame_pos,
                    'time': frame_pos / fps,
                    'frame': frame
                })
        
        cap.release()
        
        return {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'frame_samples': frame_samples,
            'resolution': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        }
    
    def select_frames_with_ai_enhancement(self, video_path: str, story_structure: Dict, target_panels: int) -> List[Dict]:
        """Select frames using AI enhancement for quality and story relevance"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        selected_frames = []
        checkpoints = story_structure['checkpoints']
        
        for i, checkpoint in enumerate(checkpoints[:target_panels]):
            target_time = checkpoint['time']
            expected_emotion = checkpoint.get('expected_emotion', 'neutral')
            story_phase = checkpoint.get('phase', 'unknown')
            
            print(f"🎯 Selecting frame {i+1}/{target_panels} - {story_phase} ({expected_emotion})")
            
            # Search window around target time
            best_frame_data = self.find_best_frame_in_window(
                cap, target_time, fps, expected_emotion, story_phase
            )
            
            if best_frame_data:
                selected_frames.append({
                    'index': i,
                    'checkpoint': checkpoint,
                    'frame_data': best_frame_data,
                    'ai_analysis': best_frame_data.get('ai_analysis', {}),
                    'quality_score': best_frame_data.get('quality_score', 0.5)
                })
        
        cap.release()
        return selected_frames
    
    def find_best_frame_in_window(self, cap, target_time: float, fps: float, expected_emotion: str, story_phase: str) -> Optional[Dict]:
        """Find best frame in time window using AI analysis"""
        search_window = 3.0  # Search ±3 seconds
        time_offsets = np.linspace(-search_window, search_window, 13)  # 13 samples
        
        best_frame = None
        best_score = 0
        
        for offset in time_offsets:
            frame_time = max(0, target_time + offset)
            frame_number = int(frame_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # AI analysis of frame
            ai_analysis = self.analyze_frame_with_ai(frame, expected_emotion, story_phase)
            
            # Calculate comprehensive score
            score = self.calculate_frame_score(ai_analysis, expected_emotion)
            
            if score > best_score:
                best_score = score
                best_frame = {
                    'frame': frame,
                    'time': frame_time,
                    'frame_number': frame_number,
                    'ai_analysis': ai_analysis,
                    'quality_score': score
                }
        
        # Save best frame
        if best_frame:
            frame_path = os.path.join(self.output_dir, f'frame{len(os.listdir(self.output_dir)) if os.path.exists(self.output_dir) else 0:03d}.png')
            os.makedirs(self.output_dir, exist_ok=True)
            cv2.imwrite(frame_path, best_frame['frame'])
            best_frame['path'] = frame_path
        
        return best_frame
    
    def analyze_frame_with_ai(self, frame: np.ndarray, expected_emotion: str, story_phase: str) -> Dict:
        """Comprehensive AI analysis of frame"""
        analysis = {
            'faces': [],
            'emotions': {},
            'quality_metrics': {},
            'story_relevance': 0.0,
            'visual_features': {}
        }
        
        try:
            # Face detection and analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4) if self.face_detector else []
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                # Emotion analysis for this face
                face_emotions = self.emotion_analyzer.analyze_face_emotion(face_roi)
                
                # Eye quality analysis
                eye_quality = self.emotion_analyzer.analyze_eye_quality(gray[y:y+h, x:x+w])
                
                analysis['faces'].append({
                    'bbox': (x, y, w, h),
                    'emotions': face_emotions,
                    'eye_quality': eye_quality,
                    'size_score': (w * h) / (frame.shape[0] * frame.shape[1])  # Relative size
                })
            
            # Overall emotion analysis
            if analysis['faces']:
                # Aggregate emotions from all faces
                all_emotions = {}
                for face in analysis['faces']:
                    for emotion, score in face['emotions'].items():
                        all_emotions[emotion] = all_emotions.get(emotion, 0) + score
                
                # Normalize
                total_emotion = sum(all_emotions.values())
                if total_emotion > 0:
                    analysis['emotions'] = {k: v/total_emotion for k, v in all_emotions.items()}
            
            # Visual quality metrics
            analysis['quality_metrics'] = {
                'sharpness': self.calculate_sharpness(gray),
                'brightness': np.mean(gray),
                'contrast': np.std(gray),
                'face_count': len(faces)
            }
            
            # Story relevance based on phase
            analysis['story_relevance'] = self.calculate_story_relevance(analysis, story_phase)
            
        except Exception as e:
            print(f"AI analysis error: {e}")
        
        return analysis
    
    def calculate_frame_score(self, ai_analysis: Dict, expected_emotion: str) -> float:
        """Calculate comprehensive frame quality score"""
        score = 0.0
        
        # Emotion matching (30%)
        emotions = ai_analysis.get('emotions', {})
        emotion_match = emotions.get(expected_emotion, 0)
        score += emotion_match * 0.3
        
        # Face quality (25%)
        faces = ai_analysis.get('faces', [])
        if faces:
            avg_eye_quality = sum(f.get('eye_quality', 0.5) for f in faces) / len(faces)
            score += avg_eye_quality * 0.25
        
        # Visual quality (25%)
        quality_metrics = ai_analysis.get('quality_metrics', {})
        sharpness = min(1.0, quality_metrics.get('sharpness', 0) / 1000)  # Normalize
        brightness = 1.0 - abs(quality_metrics.get('brightness', 128) - 128) / 128  # Prefer mid-brightness
        contrast = min(1.0, quality_metrics.get('contrast', 0) / 50)  # Normalize
        
        visual_quality = (sharpness + brightness + contrast) / 3
        score += visual_quality * 0.25
        
        # Story relevance (20%)
        story_relevance = ai_analysis.get('story_relevance', 0.5)
        score += story_relevance * 0.2
        
        return min(1.0, score)
    
    def calculate_sharpness(self, gray_image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        return cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    def calculate_story_relevance(self, analysis: Dict, story_phase: str) -> float:
        """Calculate how relevant frame is to current story phase"""
        relevance = 0.5  # Base relevance
        
        # Phase-specific relevance factors
        phase_factors = {
            'Opening': {'face_count': 1.0, 'brightness': 0.7},
            'Character Setup': {'face_count': 2.0, 'emotions': {'happy': 1.0}},
            'Conflict Introduction': {'emotions': {'surprised': 1.0, 'angry': 0.8}},
            'Rising Tension': {'emotions': {'angry': 1.0, 'sad': 0.8}},
            'Peak Conflict': {'emotions': {'angry': 1.0, 'excited': 0.8}},
            'Climax Resolution': {'emotions': {'excited': 1.0, 'happy': 0.8}},
            'Falling Action': {'emotions': {'neutral': 1.0}},
            'Resolution': {'emotions': {'happy': 1.0}, 'brightness': 0.8}
        }
        
        factors = phase_factors.get(story_phase, {})
        
        # Apply factors
        if 'face_count' in factors:
            actual_faces = len(analysis.get('faces', []))
            expected_faces = factors['face_count']
            if actual_faces >= expected_faces:
                relevance += 0.2
        
        if 'emotions' in factors:
            frame_emotions = analysis.get('emotions', {})
            for emotion, weight in factors['emotions'].items():
                emotion_score = frame_emotions.get(emotion, 0)
                relevance += emotion_score * weight * 0.3
        
        return min(1.0, relevance)
    
    def generate_editable_comic_structure(self, selected_frames: List[Dict], story_structure: Dict) -> Dict:
        """Generate comic structure with editable elements"""
        pages = []
        panels_per_page = 4
        
        for page_num in range(0, len(selected_frames), panels_per_page):
            page_frames = selected_frames[page_num:page_num + panels_per_page]
            
            page = {
                'page_number': page_num // panels_per_page + 1,
                'panels': [],
                'editable_elements': []
            }
            
            for panel_num, frame_data in enumerate(page_frames):
                panel = {
                    'panel_number': panel_num + 1,
                    'global_panel_number': page_num + panel_num + 1,
                    'frame_path': frame_data['frame_data']['path'],
                    'story_phase': frame_data['checkpoint']['phase'],
                    'ai_analysis': frame_data['ai_analysis'],
                    'editable_properties': {
                        'replaceable': True,
                        'bubble_positions': [],
                        'visual_effects': []
                    }
                }
                
                page['panels'].append(panel)
            
            pages.append(page)
        
        return {'pages': pages, 'total_panels': len(selected_frames)}
    
    def create_ai_enhanced_bubbles(self, selected_frames: List[Dict], story_structure: Dict) -> List[Dict]:
        """Create AI-enhanced speech bubbles with smart positioning"""
        bubbles = []
        
        for i, frame_data in enumerate(selected_frames):
            checkpoint = frame_data['checkpoint']
            ai_analysis = frame_data['ai_analysis']
            
            # Generate contextual text
            bubble_text = self.generate_contextual_text(checkpoint, ai_analysis)
            
            # Smart positioning based on faces
            position = self.calculate_smart_bubble_position(ai_analysis)
            
            bubble = {
                'panel_index': i,
                'text': bubble_text,
                'position': position,
                'style': self.determine_bubble_style(ai_analysis, checkpoint),
                'editable': True,
                'ai_generated': True,
                'context': {
                    'story_phase': checkpoint['phase'],
                    'expected_emotion': checkpoint.get('expected_emotion', 'neutral'),
                    'detected_emotions': ai_analysis.get('emotions', {})
                }
            }
            
            bubbles.append(bubble)
        
        return bubbles
    
    def generate_contextual_text(self, checkpoint: Dict, ai_analysis: Dict) -> str:
        """Generate contextually appropriate text based on story and AI analysis"""
        phase = checkpoint.get('phase', 'unknown')
        story_text = checkpoint.get('story_text', '')
        detected_emotions = ai_analysis.get('emotions', {})
        
        # If we have original story text, use it
        if story_text and len(story_text.strip()) > 10:
            return story_text
        
        # Generate based on phase and emotions
        dominant_emotion = max(detected_emotions.items(), key=lambda x: x[1])[0] if detected_emotions else 'neutral'
        
        contextual_texts = {
            'Opening': {
                'happy': "What a wonderful day to begin our adventure!",
                'neutral': "Our story begins in this peaceful moment.",
                'surprised': "Something unexpected is about to unfold."
            },
            'Character Setup': {
                'happy': "It's great to meet you! Let's be friends.",
                'neutral': "Let me introduce myself and explain the situation.",
                'sad': "I have something important to tell you."
            },
            'Conflict Introduction': {
                'surprised': "Wait, what's happening here?",
                'angry': "This is not what we planned!",
                'neutral': "We have a problem that needs solving."
            },
            'Rising Tension': {
                'angry': "We must fight against this challenge!",
                'sad': "Things are getting more difficult.",
                'neutral': "The situation continues to develop."
            },
            'Peak Conflict': {
                'angry': "This is our final stand!",
                'excited': "Everything depends on this moment!",
                'neutral': "The decisive moment has arrived."
            },
            'Climax Resolution': {
                'excited': "We did it! Victory is ours!",
                'happy': "Success! Our efforts have paid off.",
                'neutral': "The conflict reaches its conclusion."
            },
            'Falling Action': {
                'neutral': "Now we deal with the aftermath.",
                'sad': "We must face the consequences.",
                'happy': "Things are finally settling down."
            },
            'Resolution': {
                'happy': "What a wonderful ending to our journey!",
                'neutral': "Our story comes to a peaceful close.",
                'excited': "A new chapter begins!"
            }
        }
        
        phase_texts = contextual_texts.get(phase, {})
        return phase_texts.get(dominant_emotion, "The story continues to unfold...")
    
    def calculate_smart_bubble_position(self, ai_analysis: Dict) -> Dict:
        """Calculate smart bubble position based on face locations"""
        faces = ai_analysis.get('faces', [])
        
        if not faces:
            # Default position if no faces
            return {'x': 50, 'y': 50}
        
        # Find the largest/most prominent face
        main_face = max(faces, key=lambda f: f['size_score'])
        x, y, w, h = main_face['bbox']
        
        # Position bubble near but not overlapping face
        bubble_x = max(10, min(x + w + 20, 250))  # To the right of face
        bubble_y = max(10, y - 10)  # Slightly above face
        
        return {'x': bubble_x, 'y': bubble_y}
    
    def determine_bubble_style(self, ai_analysis: Dict, checkpoint: Dict) -> Dict:
        """Determine bubble style based on context"""
        emotions = ai_analysis.get('emotions', {})
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
        
        styles = {
            'happy': {'color': '#FFE5B4', 'border': '#FFA500', 'shape': 'round'},
            'angry': {'color': '#FFB6C1', 'border': '#FF0000', 'shape': 'jagged'},
            'sad': {'color': '#E0E6FF', 'border': '#4169E1', 'shape': 'droopy'},
            'excited': {'color': '#FFFF99', 'border': '#FFD700', 'shape': 'burst'},
            'surprised': {'color': '#F0F8FF', 'border': '#87CEEB', 'shape': 'spiky'},
            'neutral': {'color': '#FFFFFF', 'border': '#000000', 'shape': 'round'}
        }
        
        return styles.get(dominant_emotion, styles['neutral'])
    
    def assemble_final_comic(self, comic_structure: Dict, enhanced_bubbles: List[Dict]) -> Dict:
        """Assemble final comic with all enhancements"""
        final_comic = {
            'metadata': {
                'generator': 'AdvancedComicGenerator',
                'version': '2.0',
                'ai_enhanced': True,
                'editable_bubbles': True,
                'total_pages': len(comic_structure['pages']),
                'total_panels': comic_structure['total_panels']
            },
            'pages': [],
            'editable_elements': {
                'bubbles': enhanced_bubbles,
                'panels': []
            }
        }
        
        # Process each page
        for page in comic_structure['pages']:
            enhanced_page = {
                'page_number': page['page_number'],
                'panels': [],
                'bubbles': []
            }
            
            for panel in page['panels']:
                # Add panel with editable properties
                enhanced_panel = {
                    'panel_number': panel['panel_number'],
                    'global_panel_number': panel['global_panel_number'],
                    'image': os.path.basename(panel['frame_path']),
                    'story_phase': panel['story_phase'],
                    'editable': True,
                    'ai_analysis': panel['ai_analysis']
                }
                enhanced_page['panels'].append(enhanced_panel)
                
                # Add corresponding bubble
                panel_bubble = next((b for b in enhanced_bubbles if b['panel_index'] == panel['global_panel_number'] - 1), None)
                if panel_bubble:
                    enhanced_page['bubbles'].append(panel_bubble)
            
            final_comic['pages'].append(enhanced_page)
        
        return final_comic
    
    def save_comic_data(self, comic_data: Dict):
        """Save comprehensive comic data"""
        output_path = 'output/enhanced_comic_data.json'
        os.makedirs('output', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(comic_data, f, indent=2, default=str)
        
        print(f"💾 Comic data saved to {output_path}")

class EmotionAnalyzer:
    """AI-powered emotion analysis"""
    
    def __init__(self):
        self.eye_cascade = None
        self.smile_cascade = None
        self.init_classifiers()
    
    def init_classifiers(self):
        """Initialize emotion detection classifiers"""
        try:
            eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            smile_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
            
            if os.path.exists(eye_path):
                self.eye_cascade = cv2.CascadeClassifier(eye_path)
            if os.path.exists(smile_path):
                self.smile_cascade = cv2.CascadeClassifier(smile_path)
        except:
            pass
    
    def analyze_face_emotion(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Analyze emotion in face region"""
        emotions = {'happy': 0, 'sad': 0, 'angry': 0, 'surprised': 0, 'neutral': 0, 'excited': 0}
        
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            
            # Smile detection
            if self.smile_cascade:
                smiles = self.smile_cascade.detectMultiScale(gray, 1.8, 20)
                if len(smiles) > 0:
                    emotions['happy'] = min(1.0, len(smiles) * 0.4)
                    emotions['excited'] = min(1.0, len(smiles) * 0.3)
            
            # Eye analysis
            if self.eye_cascade:
                eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 3)
                if len(eyes) >= 2:
                    # Wide eyes might indicate surprise
                    avg_eye_size = sum(w * h for x, y, w, h in eyes) / len(eyes)
                    if avg_eye_size > gray.shape[0] * gray.shape[1] * 0.05:
                        emotions['surprised'] = 0.6
                elif len(eyes) < 2:
                    # Closed eyes might indicate sadness or sleepiness
                    emotions['sad'] = 0.4
            
            # Normalize emotions
            total = sum(emotions.values())
            if total == 0:
                emotions['neutral'] = 1.0
            else:
                emotions = {k: v/total for k, v in emotions.items()}
        
        except Exception as e:
            emotions['neutral'] = 1.0
        
        return emotions
    
    def analyze_eye_quality(self, face_gray: np.ndarray) -> float:
        """Analyze eye openness quality"""
        if self.eye_cascade is None:
            return 0.7
        
        try:
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3)
            
            if len(eyes) == 0:
                return 0.2  # No eyes detected
            elif len(eyes) == 1:
                return 0.6  # One eye visible
            else:
                # Analyze eye regions for openness
                eye_scores = []
                for (ex, ey, ew, eh) in eyes:
                    eye_region = face_gray[ey:ey+eh, ex:ex+ew]
                    if eye_region.size > 0:
                        variance = np.var(eye_region)
                        openness = min(1.0, variance / 400.0)
                        eye_scores.append(openness)
                
                return sum(eye_scores) / len(eye_scores) if eye_scores else 0.5
        
        except:
            return 0.5

class StoryAnalyzer:
    """AI-powered story structure analysis"""
    
    def analyze_complete_story(self, subtitles: List, video_duration: float) -> Dict:
        """Analyze complete story structure"""
        if not subtitles:
            return self.create_default_structure(video_duration)
        
        # Define story phases
        phases = [
            {'name': 'Opening', 'start': 0.0, 'end': 0.125, 'panels': 6},
            {'name': 'Character Setup', 'start': 0.125, 'end': 0.25, 'panels': 6},
            {'name': 'Conflict Introduction', 'start': 0.25, 'end': 0.375, 'panels': 6},
            {'name': 'Rising Tension', 'start': 0.375, 'end': 0.5, 'panels': 6},
            {'name': 'Peak Conflict', 'start': 0.5, 'end': 0.625, 'panels': 6},
            {'name': 'Climax Resolution', 'start': 0.625, 'end': 0.75, 'panels': 6},
            {'name': 'Falling Action', 'start': 0.75, 'end': 0.875, 'panels': 6},
            {'name': 'Resolution', 'start': 0.875, 'end': 1.0, 'panels': 6}
        ]
        
        checkpoints = []
        checkpoint_id = 0
        
        for phase in phases:
            phase_start = phase['start'] * video_duration
            phase_end = phase['end'] * video_duration
            
            # Find subtitles in this phase
            phase_subs = [s for s in subtitles if phase_start <= s.start.total_seconds() <= phase_end]
            
            # Create checkpoints for this phase
            for i in range(phase['panels']):
                checkpoint_time = phase_start + (i / phase['panels']) * (phase_end - phase_start)
                
                # Find closest subtitle
                closest_sub = None
                if phase_subs:
                    closest_sub = min(phase_subs, key=lambda s: abs(s.start.total_seconds() - checkpoint_time))
                
                checkpoints.append({
                    'id': checkpoint_id,
                    'time': checkpoint_time,
                    'phase': phase['name'],
                    'story_text': closest_sub.content if closest_sub else f"{phase['name']} continues...",
                    'expected_emotion': self.get_phase_emotion(phase['name'])
                })
                checkpoint_id += 1
        
        return {
            'phases': phases,
            'checkpoints': checkpoints,
            'video_duration': video_duration
        }
    
    def get_phase_emotion(self, phase_name: str) -> str:
        """Get expected emotion for story phase"""
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
    
    def create_default_structure(self, video_duration: float) -> Dict:
        """Create default story structure when no subtitles"""
        checkpoints = []
        for i in range(48):
            time_pos = (i / 47) * video_duration if video_duration > 0 else i * 2
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
                'id': i,
                'time': time_pos,
                'phase': phase,
                'story_text': f"The story continues in the {phase.lower()} phase.",
                'expected_emotion': self.get_phase_emotion(phase)
            })
        
        return {
            'phases': [],
            'checkpoints': checkpoints,
            'video_duration': video_duration
        }

def create_advanced_ai_comic(video_path: str, subtitles: List = None, target_panels: int = 48) -> bool:
    """
    Create advanced AI-enhanced comic with editable bubbles
    Based on automated-comic-generation-with-ai-enhancement pattern
    """
    try:
        generator = AdvancedComicGenerator()
        comic_data = generator.process_video_to_comic(video_path, subtitles, target_panels)
        
        print("🎉 Advanced AI Comic Generation Complete!")
        print(f"📊 Generated: {comic_data['metadata']['total_pages']} pages, {comic_data['metadata']['total_panels']} panels")
        print(f"🧠 AI Enhanced: {comic_data['metadata']['ai_enhanced']}")
        print(f"✏️ Editable Bubbles: {comic_data['metadata']['editable_bubbles']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced comic generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False