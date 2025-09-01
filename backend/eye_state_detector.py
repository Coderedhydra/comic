"""
Eye State Detection and Frame Selection
Ensures selected frames have open eyes
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Dict
import os

class EyeStateDetector:
    """Detect if eyes are open or closed in frames"""
    
    def __init__(self):
        """Initialize eye detection"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Try to use MediaPipe for better accuracy
        self.use_mediapipe = False
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
            print("✅ Using MediaPipe for accurate eye detection")
        except:
            print("⚠️ MediaPipe not available, using OpenCV")
    
    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR) to detect if eye is open"""
        # EAR = (vertical_distance) / (horizontal_distance)
        # For open eyes: EAR > 0.2
        # For closed eyes: EAR < 0.2
        
        if len(eye_points) < 6:
            return 0.3  # Default to open
            
        # Calculate distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear
    
    def detect_eyes_mediapipe(self, image):
        """Detect eyes using MediaPipe"""
        if not self.use_mediapipe:
            return []
            
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        eye_states = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Left eye indices
                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                # Right eye indices  
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]
                
                h, w = image.shape[:2]
                
                # Get left eye points
                left_eye_points = []
                for idx in LEFT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    left_eye_points.append([x, y])
                
                # Get right eye points
                right_eye_points = []
                for idx in RIGHT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    right_eye_points.append([x, y])
                
                # Calculate EAR for both eyes
                left_ear = self.calculate_eye_aspect_ratio(np.array(left_eye_points))
                right_ear = self.calculate_eye_aspect_ratio(np.array(right_eye_points))
                
                # Average EAR
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Determine if eyes are open (threshold: 0.2)
                is_open = avg_ear > 0.2
                
                eye_states.append({
                    'left_ear': left_ear,
                    'right_ear': right_ear,
                    'avg_ear': avg_ear,
                    'eyes_open': is_open
                })
                
        return eye_states
    
    def detect_eyes_opencv(self, image):
        """Fallback eye detection using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        eye_states = []
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            
            # Simple heuristic: if we detect 2 eyes, they're likely open
            # If we detect 0 or 1, they might be closed
            eyes_open = len(eyes) >= 2
            
            # Additional check: analyze eye regions for darkness
            if len(eyes) > 0:
                eye_openness = []
                for (ex, ey, ew, eh) in eyes[:2]:  # Check first 2 eyes
                    eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                    # Calculate variance - open eyes have more variance
                    variance = np.var(eye_roi)
                    eye_openness.append(variance > 100)  # Threshold
                
                eyes_open = any(eye_openness) if eye_openness else False
            
            eye_states.append({
                'num_eyes_detected': len(eyes),
                'eyes_open': eyes_open
            })
            
        return eye_states
    
    def are_eyes_open(self, image_path):
        """Check if eyes are open in the image"""
        image = cv2.imread(image_path)
        if image is None:
            return True  # Default to true if can't read
        
        # Try MediaPipe first
        if self.use_mediapipe:
            eye_states = self.detect_eyes_mediapipe(image)
            if eye_states:
                # Return true if any face has open eyes
                return any(state['eyes_open'] for state in eye_states)
        
        # Fallback to OpenCV
        eye_states = self.detect_eyes_opencv(image)
        if eye_states:
            return any(state['eyes_open'] for state in eye_states)
        
        # If no faces detected, assume eyes are open
        return True
    
    def score_frame_quality(self, image_path):
        """Score frame quality based on eye state and other factors"""
        image = cv2.imread(image_path)
        if image is None:
            return 0.0
        
        score = 0.0
        
        # 1. Eye state score (most important)
        if self.are_eyes_open(image_path):
            score += 50.0
        
        # 2. Face detection score
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            score += 20.0
        
        # 3. Image sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        if sharpness > 100:
            score += 15.0
        
        # 4. Brightness (not too dark or bright)
        brightness = np.mean(gray)
        if 50 < brightness < 200:
            score += 10.0
        
        # 5. Contrast
        contrast = np.std(gray)
        if contrast > 30:
            score += 5.0
        
        return score

class SmartFrameSelector:
    """Select best frames avoiding closed eyes"""
    
    def __init__(self):
        self.eye_detector = EyeStateDetector()
    
    def select_best_frames(self, frame_paths: List[str], num_frames: int = 16) -> List[str]:
        """Select best frames based on eye state and quality"""
        print("👁️ Analyzing frames for open eyes...")
        
        # Score all frames
        frame_scores = []
        for i, frame_path in enumerate(frame_paths):
            score = self.eye_detector.score_frame_quality(frame_path)
            frame_scores.append((frame_path, score))
            
            # Debug info
            if i % 10 == 0:
                eyes_open = self.eye_detector.are_eyes_open(frame_path)
                print(f"  Frame {i}: Score={score:.1f}, Eyes={'Open' if eyes_open else 'Closed'}")
        
        # Sort by score (highest first)
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top frames with good distribution
        selected_frames = []
        selected_indices = set()
        
        # First pass: get frames with open eyes
        for frame_path, score in frame_scores:
            if len(selected_frames) >= num_frames:
                break
                
            # Get frame index
            frame_name = os.path.basename(frame_path)
            frame_idx = int(''.join(filter(str.isdigit, frame_name)))
            
            # Ensure some spacing between frames
            too_close = any(abs(frame_idx - idx) < 5 for idx in selected_indices)
            
            if not too_close and score > 30:  # Minimum quality threshold
                selected_frames.append(frame_path)
                selected_indices.add(frame_idx)
        
        # If not enough frames, add more with relaxed criteria
        if len(selected_frames) < num_frames:
            for frame_path, score in frame_scores:
                if frame_path not in selected_frames:
                    selected_frames.append(frame_path)
                    if len(selected_frames) >= num_frames:
                        break
        
        # Sort selected frames by name to maintain order
        selected_frames.sort()
        
        print(f"✅ Selected {len(selected_frames)} frames with open eyes")
        return selected_frames[:num_frames]

def enhance_frame_selection(frames_dir: str = 'frames', output_dir: str = 'frames/final'):
    """Enhance frame selection to avoid closed eyes"""
    import shutil
    
    # Get all frames
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
    
    if not frame_paths:
        print("❌ No frames found")
        return
    
    # Select best frames
    selector = SmartFrameSelector()
    best_frames = selector.select_best_frames(frame_paths, num_frames=16)
    
    # Copy selected frames to output
    os.makedirs(output_dir, exist_ok=True)
    
    for i, frame_path in enumerate(best_frames):
        output_path = os.path.join(output_dir, f'frame{i:03d}.png')
        shutil.copy2(frame_path, output_path)
        print(f"  Copied: {os.path.basename(frame_path)} → {os.path.basename(output_path)}")
    
    print(f"✅ Enhanced frame selection complete: {len(best_frames)} frames")

# Integration with existing keyframe generation
def generate_keyframes_with_eye_check(video_path: str, num_frames: int = 16):
    """Generate keyframes ensuring eyes are open"""
    # First extract more frames than needed
    from backend.keyframes.extract_frames import extract_frames
    
    print("🎬 Extracting frames from video...")
    extract_frames(video_path, num_frames=num_frames * 3)  # Extract 3x frames
    
    # Then select best frames with open eyes
    enhance_frame_selection('frames', 'frames/final')

if __name__ == "__main__":
    # Test the eye detection
    detector = EyeStateDetector()
    
    # Test on a sample image
    test_image = "frames/frame001.png"
    if os.path.exists(test_image):
        eyes_open = detector.are_eyes_open(test_image)
        print(f"Eyes open in {test_image}: {eyes_open}")
        
        score = detector.score_frame_quality(test_image)
        print(f"Frame quality score: {score}")