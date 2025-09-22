"""
Complete Story Builder for Comic Summarization
Builds a coherent, complete story that makes sense from beginning to end
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple
import srt
from datetime import timedelta
import re

class CompleteStoryBuilder:
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.init_opencv_classifiers()
        
    def init_opencv_classifiers(self):
        """Initialize OpenCV classifiers"""
        try:
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            
            if os.path.exists(face_cascade_path):
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
                print("✅ Face detection ready")
            
            if os.path.exists(eye_cascade_path):
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
                print("✅ Eye detection ready")
                
        except Exception as e:
            print(f"⚠️ OpenCV setup: {e}")
    
    def analyze_subtitle_importance(self, text: str, position_in_video: float) -> float:
        """
        Analyze how important this subtitle is for story understanding
        position_in_video: 0.0 = start, 1.0 = end
        """
        if not text:
            return 0.1
        
        text_lower = text.lower()
        importance = 0.0
        
        # Story structure importance based on position
        if position_in_video < 0.1:  # Opening 10%
            opening_words = ['meet', 'introduce', 'begin', 'start', 'welcome', 'hello', 'first', 'once']
            importance += sum(0.5 for word in opening_words if word in text_lower)
        elif position_in_video > 0.9:  # Ending 10%
            ending_words = ['end', 'finish', 'final', 'last', 'goodbye', 'farewell', 'conclusion', 'over']
            importance += sum(0.5 for word in ending_words if word in text_lower)
        
        # High importance story elements
        story_keywords = {
            'character_intro': ['name', 'called', 'meet', 'this is', 'i am', 'my name'],
            'conflict': ['problem', 'trouble', 'danger', 'enemy', 'fight', 'battle', 'against'],
            'emotion': ['love', 'hate', 'angry', 'sad', 'happy', 'afraid', 'excited'],
            'action': ['go', 'run', 'escape', 'chase', 'attack', 'defend', 'save', 'help'],
            'revelation': ['discover', 'find', 'reveal', 'truth', 'secret', 'realize', 'understand'],
            'relationship': ['friend', 'family', 'father', 'mother', 'brother', 'sister', 'together'],
            'goal': ['must', 'need', 'want', 'goal', 'mission', 'quest', 'journey', 'destination'],
            'climax': ['final', 'last', 'ultimate', 'decide', 'choice', 'fate', 'destiny']
        }
        
        for category, words in story_keywords.items():
            matches = sum(1 for word in words if word in text_lower)
            importance += matches * 0.3
        
        # Boost for dialogue vs narration
        if '"' in text or "'" in text:
            importance += 0.2
        
        # Boost for questions and exclamations (usually important)
        if '?' in text:
            importance += 0.3
        if '!' in text:
            importance += 0.2
        
        # Penalty for very short or filler text
        if len(text.strip()) < 10:
            importance *= 0.5
        
        filler_words = ['um', 'uh', 'well', 'so', 'like', 'you know']
        if any(word in text_lower for word in filler_words):
            importance *= 0.7
        
        return min(1.0, importance)
    
    def build_complete_story_structure(self, subtitles: List, target_panels: int = 48) -> List[Dict]:
        """
        Build a complete, coherent story structure covering the entire video
        """
        print(f"📖 Building complete story structure for {target_panels} panels...")
        
        if not subtitles:
            return []
        
        # Sort subtitles chronologically
        subtitles.sort(key=lambda x: x.start.total_seconds())
        
        video_duration = subtitles[-1].end.total_seconds()
        print(f"🎬 Video duration: {video_duration:.1f} seconds")
        
        # Analyze all subtitles for story importance
        analyzed_subtitles = []
        for i, subtitle in enumerate(subtitles):
            position = i / (len(subtitles) - 1) if len(subtitles) > 1 else 0
            time_position = subtitle.start.total_seconds() / video_duration
            
            importance = self.analyze_subtitle_importance(subtitle.content, time_position)
            
            analyzed_subtitles.append({
                'subtitle': subtitle,
                'index': i,
                'time': subtitle.start.total_seconds(),
                'importance': importance,
                'position': time_position,
                'text': subtitle.content
            })
        
        # Create story structure with proper narrative flow
        story_panels = []
        
        # Define story phases with specific panel allocation
        story_phases = [
            {'name': 'Opening', 'start': 0.0, 'end': 0.15, 'panels': 7},      # Panels 1-7
            {'name': 'Setup', 'start': 0.15, 'end': 0.25, 'panels': 5},       # Panels 8-12
            {'name': 'Rising Action', 'start': 0.25, 'end': 0.50, 'panels': 12}, # Panels 13-24
            {'name': 'Climax', 'start': 0.50, 'end': 0.75, 'panels': 12},     # Panels 25-36
            {'name': 'Falling Action', 'start': 0.75, 'end': 0.90, 'panels': 7}, # Panels 37-43
            {'name': 'Resolution', 'start': 0.90, 'end': 1.0, 'panels': 5}     # Panels 44-48
        ]
        
        panel_number = 1
        
        for phase in story_phases:
            print(f"📍 Building {phase['name']} phase ({phase['panels']} panels)")
            
            # Find subtitles in this phase
            phase_subtitles = [
                sub for sub in analyzed_subtitles
                if phase['start'] <= sub['position'] <= phase['end']
            ]
            
            if not phase_subtitles:
                # If no subtitles in phase, create narrative bridges
                phase_duration = (phase['end'] - phase['start']) * video_duration
                for i in range(phase['panels']):
                    panel_time = phase['start'] * video_duration + (i / phase['panels']) * phase_duration
                    
                    story_panels.append({
                        'panel': panel_number,
                        'time': panel_time,
                        'phase': phase['name'],
                        'text': self.create_narrative_bridge(phase['name'], i, phase['panels']),
                        'importance': 0.5,
                        'is_bridge': True
                    })
                    panel_number += 1
                continue
            
            # Sort phase subtitles by importance, then by time
            phase_subtitles.sort(key=lambda x: (-x['importance'], x['time']))
            
            # Distribute panels across this phase
            panels_needed = phase['panels']
            
            if len(phase_subtitles) >= panels_needed:
                # Take the most important subtitles
                selected = phase_subtitles[:panels_needed]
                # Sort back to chronological order
                selected.sort(key=lambda x: x['time'])
            else:
                # Use all available subtitles and fill gaps
                selected = sorted(phase_subtitles, key=lambda x: x['time'])
                
                # Fill remaining panels with interpolated moments
                while len(selected) < panels_needed:
                    # Find largest time gap and add a moment there
                    if len(selected) == 0:
                        # Add at phase start
                        gap_time = phase['start'] * video_duration
                    else:
                        # Find largest gap between consecutive subtitles
                        gaps = []
                        for i in range(len(selected) - 1):
                            gap_size = selected[i+1]['time'] - selected[i]['time']
                            gaps.append((gap_size, i))
                        
                        if gaps:
                            largest_gap = max(gaps)
                            gap_index = largest_gap[1]
                            gap_time = (selected[gap_index]['time'] + selected[gap_index + 1]['time']) / 2
                        else:
                            gap_time = selected[-1]['time'] + 10  # Add after last
                    
                    # Create interpolated moment
                    selected.append({
                        'time': gap_time,
                        'text': self.create_narrative_bridge(phase['name'], len(selected), panels_needed),
                        'importance': 0.4,
                        'is_bridge': True
                    })
                    
                    selected.sort(key=lambda x: x['time'])
            
            # Create story panels for this phase
            for moment in selected[:panels_needed]:
                story_panels.append({
                    'panel': panel_number,
                    'time': moment['time'],
                    'phase': phase['name'],
                    'text': moment['text'],
                    'importance': moment['importance'],
                    'is_bridge': moment.get('is_bridge', False)
                })
                panel_number += 1
        
        print(f"✅ Built complete story with {len(story_panels)} panels")
        print(f"📚 Story phases: Opening→Setup→Rising Action→Climax→Falling Action→Resolution")
        
        return story_panels
    
    def create_narrative_bridge(self, phase: str, index: int, total: int) -> str:
        """Create narrative bridge text for story continuity"""
        
        bridges = {
            'Opening': [
                "Our story begins in a world where adventure awaits.",
                "We meet our main character as their journey starts.",
                "The setting comes alive with rich detail and atmosphere.",
                "Initial circumstances set the stage for what's to come.",
                "Characters are introduced with their unique personalities.",
                "The world-building establishes the story's foundation.",
                "Early events hint at the greater adventure ahead."
            ],
            'Setup': [
                "The story's central conflict begins to take shape.",
                "Characters discover their roles in the unfolding drama.",
                "Important relationships are established and developed.",
                "The main quest or challenge becomes clear to everyone.",
                "Stakes are established as characters commit to their path."
            ],
            'Rising Action': [
                "Challenges intensify as our heroes face greater obstacles.",
                "Character development deepens through trials and conflicts.",
                "The antagonist's power and motivation become clearer.",
                "Alliances form and shift as the situation grows complex.",
                "Personal stakes intertwine with the larger conflict.",
                "Each victory comes with new challenges and complications.",
                "Trust is tested as characters face difficult choices.",
                "The scope of the conflict expands beyond initial expectations.",
                "Characters must adapt their strategies to survive.",
                "Relationships are strained by the mounting pressure.",
                "New allies and enemies emerge to complicate matters.",
                "The path forward becomes increasingly uncertain."
            ],
            'Climax': [
                "The final confrontation begins as all forces converge.",
                "Everything our heroes learned is put to the ultimate test.",
                "The conflict reaches its most intense and dangerous point.",
                "Characters must overcome their deepest fears and flaws.",
                "Truth is finally revealed about the central mystery.",
                "Sacrifices must be made as the stakes reach their peak.",
                "The fate of everyone hangs in the balance.",
                "Heroes and villains clash in the most dramatic moments.",
                "Unexpected twists change the nature of the battle.",
                "Characters discover strength they never knew they had.",
                "The outcome will determine the future for everyone.",
                "This is the moment everything has been building toward."
            ],
            'Falling Action': [
                "The immediate crisis is resolved through heroic action.",
                "Characters begin to process what they've experienced.",
                "The world starts to heal from the conflict's aftermath.",
                "Relationships are redefined by shared trials.",
                "Consequences of choices become clear to everyone.",
                "Order begins to emerge from the chaos of battle.",
                "Characters reflect on their growth and transformation."
            ],
            'Resolution': [
                "Peace is restored and the world finds new balance.",
                "Characters have grown and changed through their journey.",
                "Loose ends are tied up and questions are answered.",
                "The community celebrates the heroes' achievements.",
                "New beginnings emerge from the resolution of conflict."
            ]
        }
        
        phase_bridges = bridges.get(phase, ["The story continues to unfold."])
        return phase_bridges[index % len(phase_bridges)]
    
    def check_eye_quality(self, image_path: str) -> float:
        """Check eye quality, but don't drop frames - just score them"""
        try:
            if not self.face_cascade or not self.eye_cascade:
                return 0.8  # Default good score
            
            img = cv2.imread(image_path)
            if img is None:
                return 0.5
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return 0.6  # No face, but don't penalize too much
            
            best_score = 0.0
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+int(h*0.6), x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                
                if len(eyes) >= 2:
                    # Good - both eyes detected
                    eye_scores = []
                    for (ex, ey, ew, eh) in eyes:
                        eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                        if eye_region.size > 0:
                            variance = np.var(eye_region)
                            openness = min(1.0, variance / 600.0)  # More lenient
                            eye_scores.append(openness)
                    
                    if eye_scores:
                        avg_score = sum(eye_scores) / len(eye_scores)
                        best_score = max(best_score, avg_score)
                elif len(eyes) == 1:
                    best_score = max(best_score, 0.7)  # One eye visible
                else:
                    best_score = max(best_score, 0.4)  # No eyes, but don't drop
            
            return best_score
            
        except Exception as e:
            return 0.6  # Default middle score on error
    
    def generate_story_frames(self, video_path: str, story_panels: List[Dict], output_dir: str = 'frames/final') -> bool:
        """Generate frames for story panels - select best frame near each time, don't drop"""
        print(f"🎬 Generating frames for {len(story_panels)} story panels...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear existing frames
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(output_dir, f))
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        successful_frames = []
        
        for i, panel in enumerate(story_panels):
            target_time = panel['time']
            
            # Try multiple frames around target time to find best quality
            time_offsets = [0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5]
            best_frame = None
            best_score = 0
            
            for offset in time_offsets:
                frame_time = max(0, target_time + offset)
                frame_number = int(frame_time * fps)
                
                if frame_number >= total_frames:
                    continue
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Save temp frame for quality check
                temp_path = f'temp_quality_{i}.png'
                cv2.imwrite(temp_path, frame)
                
                # Check quality (eyes + general image quality)
                eye_score = self.check_eye_quality(temp_path)
                
                # Add other quality factors
                total_score = eye_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_frame = frame.copy()
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Always save a frame - never drop panels
            if best_frame is not None:
                output_path = os.path.join(output_dir, f'frame{i:03d}.png')
                cv2.imwrite(output_path, best_frame)
                successful_frames.append({
                    'panel': i + 1,
                    'time': target_time,
                    'phase': panel['phase'],
                    'quality_score': best_score
                })
                print(f"✅ Panel {i+1:2d} ({panel['phase']:12s}): {panel['text'][:40]}... (quality: {best_score:.2f})")
            else:
                # Fallback - grab any frame near target time
                frame_number = int(target_time * fps)
                frame_number = max(0, min(frame_number, total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    output_path = os.path.join(output_dir, f'frame{i:03d}.png')
                    cv2.imwrite(output_path, frame)
                    successful_frames.append({
                        'panel': i + 1,
                        'time': target_time,
                        'phase': panel['phase'],
                        'quality_score': 0.3
                    })
                    print(f"📸 Panel {i+1:2d} ({panel['phase']:12s}): {panel['text'][:40]}... (fallback)")
        
        cap.release()
        
        print(f"🎉 Generated {len(successful_frames)} story frames")
        
        # Save metadata
        with open(os.path.join(output_dir, 'story_metadata.json'), 'w') as f:
            json.dump({
                'panels': story_panels,
                'frames': successful_frames
            }, f, indent=2, default=str)
        
        return len(successful_frames) > 0

def create_complete_story_comic(video_path: str, subtitles: List, target_panels: int = 48) -> bool:
    """
    Create a complete, coherent story comic covering entire video
    """
    builder = CompleteStoryBuilder()
    
    # Build complete story structure
    story_panels = builder.build_complete_story_structure(subtitles, target_panels)
    
    if not story_panels:
        return False
    
    # Generate frames for story
    success = builder.generate_story_frames(video_path, story_panels)
    
    return success