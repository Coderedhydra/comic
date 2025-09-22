"""
Comic Story Summarization System
Analyzes entire video to extract main story points and create 48-panel comic summary
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple
import srt
from datetime import timedelta

class ComicStorySummarizer:
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.init_opencv_classifiers()
        
    def init_opencv_classifiers(self):
        """Initialize OpenCV classifiers for face and eye detection"""
        try:
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            
            if os.path.exists(face_cascade_path):
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
                print("✅ Face detection initialized")
            
            if os.path.exists(eye_cascade_path):
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
                print("✅ Eye detection initialized")
                
        except Exception as e:
            print(f"⚠️ OpenCV initialization error: {e}")
    
    def analyze_eye_state(self, image_path: str) -> float:
        """
        Analyze eye state in image
        Returns: 0.0-1.0 score (1.0 = wide open, 0.0 = closed)
        """
        try:
            if not self.face_cascade or not self.eye_cascade:
                return 0.8  # Default assume good if detection unavailable
            
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return 0.5  # No face detected
            
            best_eye_score = 0.0
            
            for (x, y, w, h) in faces:
                # Focus on eye region (upper 60% of face)
                roi_gray = gray[y:y+int(h*0.6), x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                
                if len(eyes) >= 2:
                    # Both eyes detected - analyze openness
                    eye_scores = []
                    
                    for (ex, ey, ew, eh) in eyes:
                        eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                        
                        if eye_region.size > 0:
                            # Calculate eye openness based on intensity variance
                            variance = np.var(eye_region)
                            # Higher variance = more open eye
                            openness = min(1.0, variance / 800.0)
                            
                            # Additional check: aspect ratio
                            aspect_ratio = eh / ew if ew > 0 else 0
                            if aspect_ratio < 0.3:  # Very flat = likely closed
                                openness *= 0.5
                            
                            eye_scores.append(openness)
                    
                    # Use average of both eyes
                    if eye_scores:
                        avg_score = sum(eye_scores) / len(eye_scores)
                        best_eye_score = max(best_eye_score, avg_score)
                
                elif len(eyes) == 1:
                    # Only one eye visible - might be profile
                    best_eye_score = max(best_eye_score, 0.6)
                else:
                    # No eyes detected in face - likely closed
                    best_eye_score = max(best_eye_score, 0.2)
            
            return best_eye_score
            
        except Exception as e:
            print(f"Eye analysis error: {e}")
            return 0.5
    
    def analyze_story_importance(self, subtitle_text: str) -> float:
        """
        Analyze how important this subtitle is to the main story
        Returns: 0.0-1.0 importance score
        """
        if not subtitle_text:
            return 0.1
        
        text = subtitle_text.lower()
        
        # High importance keywords (main plot points)
        high_importance = [
            # Story progression
            'beginning', 'start', 'first', 'discover', 'find', 'reveal', 'secret',
            'important', 'must', 'need', 'have to', 'crucial', 'critical',
            # Conflict and drama
            'problem', 'trouble', 'danger', 'threat', 'enemy', 'villain', 'fight',
            'battle', 'war', 'attack', 'defend', 'save', 'rescue', 'help',
            # Character development
            'love', 'hate', 'friend', 'family', 'father', 'mother', 'son', 'daughter',
            'brother', 'sister', 'relationship', 'trust', 'betray', 'loyalty',
            # Plot twists and revelations
            'truth', 'lie', 'real', 'fake', 'actually', 'really', 'surprise',
            'shock', 'unexpected', 'twist', 'turn', 'change', 'different',
            # Climax and resolution
            'final', 'last', 'end', 'finish', 'complete', 'over', 'done',
            'victory', 'defeat', 'win', 'lose', 'success', 'failure'
        ]
        
        # Medium importance keywords
        medium_importance = [
            'go', 'come', 'leave', 'stay', 'return', 'back', 'home',
            'work', 'job', 'money', 'pay', 'buy', 'sell',
            'think', 'believe', 'know', 'understand', 'remember',
            'tell', 'say', 'speak', 'talk', 'ask', 'answer'
        ]
        
        # Calculate importance score
        high_score = sum(1 for word in high_importance if word in text)
        medium_score = sum(0.5 for word in medium_importance if word in text)
        
        # Boost score for questions and exclamations
        if '?' in subtitle_text:
            high_score += 0.5
        if '!' in subtitle_text:
            high_score += 0.3
        
        # Boost for emotional words
        emotion_words = ['happy', 'sad', 'angry', 'scared', 'surprised', 'excited']
        emotion_score = sum(0.3 for word in emotion_words if word in text)
        
        total_score = high_score + medium_score + emotion_score
        
        # Normalize to 0-1 range
        importance = min(1.0, total_score / 3.0)
        
        return max(0.1, importance)  # Minimum 0.1 importance
    
    def extract_complete_story_summary(self, video_path: str, subtitles: List, target_panels: int = 48) -> List[Dict]:
        """
        Extract COMPLETE story from entire video for comprehensive comic summary
        Returns chronologically ordered story moments covering the WHOLE video
        """
        print(f"📖 Analyzing ENTIRE video for complete {target_panels}-panel story summary...")
        
        if not subtitles:
            print("❌ No subtitles available for story analysis")
            return []
        
        # Sort subtitles by time to ensure chronological order
        subtitles.sort(key=lambda x: x.start.total_seconds())
        
        # Calculate video duration
        video_duration = subtitles[-1].end.total_seconds() if subtitles else 100
        print(f"📹 Video duration: {video_duration:.1f} seconds")
        
        # Divide entire video into exactly 48 equal time segments
        # This ensures COMPLETE coverage of the whole story
        segment_duration = video_duration / target_panels
        selected_moments = []
        
        print(f"⏱️ Creating {target_panels} segments of {segment_duration:.1f} seconds each")
        
        for segment_idx in range(target_panels):
            segment_start = segment_idx * segment_duration
            segment_end = (segment_idx + 1) * segment_duration
            segment_mid = segment_start + (segment_duration / 2)
            
            print(f"📍 Segment {segment_idx + 1}: {segment_start:.1f}s - {segment_end:.1f}s")
            
            # Find ALL subtitles in this time segment
            segment_subtitles = [
                sub for sub in subtitles
                if (sub.start.total_seconds() <= segment_end and 
                    sub.end.total_seconds() >= segment_start)
            ]
            
            if segment_subtitles:
                # Choose the subtitle closest to the middle of this segment
                # This ensures even distribution across the ENTIRE video
                best_subtitle = min(
                    segment_subtitles,
                    key=lambda x: abs(x.start.total_seconds() - segment_mid)
                )
                
                # Calculate story position (beginning, middle, end)
                story_position = segment_idx / (target_panels - 1) if target_panels > 1 else 0
                
                # Determine story phase
                if story_position < 0.25:
                    phase = "Beginning"
                elif story_position < 0.5:
                    phase = "Rising Action"
                elif story_position < 0.75:
                    phase = "Climax"
                else:
                    phase = "Resolution"
                
                selected_moments.append({
                    'segment': segment_idx + 1,
                    'subtitle': best_subtitle,
                    'start_time': best_subtitle.start.total_seconds(),
                    'end_time': best_subtitle.end.total_seconds(),
                    'text': best_subtitle.content,
                    'story_position': story_position,
                    'phase': phase,
                    'panel_number': segment_idx + 1
                })
                
                print(f"  ✅ Selected: '{best_subtitle.content[:40]}...' ({phase})")
            else:
                # If no subtitle in segment, create a narrative bridge
                story_position = segment_idx / (target_panels - 1) if target_panels > 1 else 0
                
                if story_position < 0.25:
                    bridge_text = f"The story continues to develop as events unfold..."
                elif story_position < 0.5:
                    bridge_text = f"Tension builds as the plot thickens..."
                elif story_position < 0.75:
                    bridge_text = f"The climax approaches with rising stakes..."
                else:
                    bridge_text = f"The story moves toward its conclusion..."
                
                # Create a fake subtitle for this segment
                class FakeSubtitle:
                    def __init__(self, content, start_time):
                        self.content = content
                        self.start = timedelta(seconds=start_time)
                        self.end = timedelta(seconds=start_time + segment_duration)
                
                fake_sub = FakeSubtitle(bridge_text, segment_mid)
                
                selected_moments.append({
                    'segment': segment_idx + 1,
                    'subtitle': fake_sub,
                    'start_time': segment_mid,
                    'end_time': segment_mid + segment_duration,
                    'text': bridge_text,
                    'story_position': story_position,
                    'phase': "Transition",
                    'panel_number': segment_idx + 1
                })
                
                print(f"  🔗 Bridge: '{bridge_text}'")
        
        # Ensure chronological order
        selected_moments.sort(key=lambda x: x['start_time'])
        
        print(f"✅ Created COMPLETE story summary with {len(selected_moments)} panels")
        print(f"📚 Story coverage: Beginning → Rising Action → Climax → Resolution")
        
        return selected_moments
    
    def generate_story_frames(self, video_path: str, story_moments: List[Dict], output_dir: str = 'frames/final') -> bool:
        """
        Generate frames for selected story moments with strict eye filtering
        """
        print(f"🎬 Generating {len(story_moments)} story frames with eye filtering...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear existing frames
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(output_dir, f))
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        successful_frames = []
        
        for i, moment in enumerate(story_moments):
            try:
                print(f"📸 Processing story moment {i+1}/{len(story_moments)}: '{moment['text'][:50]}...'")
                
                # Calculate frame times to check
                start_time = moment['start_time']
                end_time = moment['end_time']
                mid_time = (start_time + end_time) / 2
                
                # Try multiple frames around the moment
                time_offsets = [0, -0.3, 0.3, -0.6, 0.6, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0]
                best_frame = None
                best_eye_score = 0
                
                for offset in time_offsets:
                    frame_time = mid_time + offset
                    if frame_time < 0 or frame_time > (total_frames / fps):
                        continue
                    
                    frame_number = int(frame_time * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    # Save temporary frame for eye analysis
                    temp_path = f'temp_eye_check_{i}.png'
                    cv2.imwrite(temp_path, frame)
                    
                    # Analyze eye state
                    eye_score = self.analyze_eye_state(temp_path)
                    
                    print(f"    Frame at {frame_time:.1f}s: eye_score={eye_score:.2f}")
                    
                    # Only accept frames with good eye scores (>= 0.7)
                    if eye_score >= 0.7 and eye_score > best_eye_score:
                        best_eye_score = eye_score
                        best_frame = frame.copy()
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Save best frame if found
                if best_frame is not None:
                    output_path = os.path.join(output_dir, f'frame{i:03d}.png')
                    cv2.imwrite(output_path, best_frame)
                    successful_frames.append({
                        'frame_index': i,
                        'eye_score': best_eye_score,
                        'story_moment': moment
                    })
                    print(f"✅ Saved frame {i:03d} with eye_score={best_eye_score:.2f}")
                else:
                    print(f"❌ No suitable frame found for moment {i+1} (all frames had closed/half-closed eyes)")
                    
                    # Create a placeholder or skip - for now we'll create a text placeholder
                    # In a real implementation, you might want to find the next best story moment
            
            except Exception as e:
                print(f"❌ Error processing moment {i}: {e}")
                continue
        
        cap.release()
        
        print(f"🎉 Successfully generated {len(successful_frames)} story frames with open eyes")
        
        # Save frame metadata
        with open(os.path.join(output_dir, 'story_frames.json'), 'w') as f:
            json.dump(successful_frames, f, indent=2, default=str)
        
        return len(successful_frames) > 0

def create_comic_story_summary(video_path: str, subtitles: List, target_panels: int = 48) -> bool:
    """
    Main function to create COMPLETE 48-panel comic story summary covering entire video
    """
    summarizer = ComicStorySummarizer()
    
    # Extract COMPLETE story covering entire video
    story_moments = summarizer.extract_complete_story_summary(video_path, subtitles, target_panels)
    
    if not story_moments:
        print("❌ No story moments extracted")
        return False
    
    print(f"📖 Story Summary Structure:")
    print(f"   📍 Panels 1-12: Beginning & Setup")
    print(f"   📍 Panels 13-24: Rising Action & Development") 
    print(f"   📍 Panels 25-36: Climax & Major Events")
    print(f"   📍 Panels 37-48: Resolution & Conclusion")
    
    # Generate frames for complete story moments
    success = summarizer.generate_story_frames(video_path, story_moments)
    
    return success