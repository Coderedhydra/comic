"""
Enhanced eye state detection to avoid half-closed eyes in frames
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List
import os

class EyeStateDetector:
    """Detect eye states (open, closed, half-closed) in images"""
    
    def __init__(self):
        # Load cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Eye aspect ratio thresholds
        self.EAR_THRESHOLD_CLOSED = 0.2
        self.EAR_THRESHOLD_HALF = 0.25
        self.EAR_THRESHOLD_OPEN = 0.3
        
    def check_eyes_state(self, image_path: str) -> Dict[str, any]:
        """
        Check the state of eyes in an image
        
        Returns:
            dict: {
                'state': 'open'|'closed'|'half_closed'|'unknown',
                'confidence': float (0-1),
                'suitable_for_comic': bool,
                'eye_aspect_ratio': float
            }
        """
        img = cv2.imread(image_path)
        if img is None:
            return {
                'state': 'unknown',
                'confidence': 0.0,
                'suitable_for_comic': False,
                'eye_aspect_ratio': 0.0
            }
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return {
                'state': 'unknown',
                'confidence': 0.0,
                'suitable_for_comic': True,  # No face, might be background
                'eye_aspect_ratio': 0.0
            }
        
        # Process the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect eyes in face region
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.05, 5)
        
        if len(eyes) < 2:
            # Less than 2 eyes detected - might be closed or profile view
            return {
                'state': 'possibly_closed',
                'confidence': 0.5,
                'suitable_for_comic': False,
                'eye_aspect_ratio': 0.0
            }
        
        # Calculate eye metrics
        eye_metrics = self._analyze_eye_openness(face_roi, eyes)
        
        # Determine state
        state, confidence, suitable = self._determine_eye_state(eye_metrics)
        
        return {
            'state': state,
            'confidence': confidence,
            'suitable_for_comic': suitable,
            'eye_aspect_ratio': eye_metrics['average_ear']
        }
    
    def _analyze_eye_openness(self, face_roi, eyes) -> Dict[str, float]:
        """Analyze how open the eyes are"""
        eye_aspects = []
        
        for (ex, ey, ew, eh) in eyes[:2]:  # Process first two eyes
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            
            # Calculate eye aspect ratio (simplified)
            # In a real implementation, we'd use facial landmarks
            # Here we use a simpler approach based on eye region intensity
            
            # Check vertical gradient (open eyes have more gradient)
            gradient = cv2.Sobel(eye_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.abs(gradient).mean()
            
            # Check darkness ratio (closed eyes are darker)
            mean_intensity = eye_roi.mean()
            
            # Estimate eye aspect ratio
            ear = self._estimate_ear(gradient_magnitude, mean_intensity, eh)
            eye_aspects.append(ear)
        
        return {
            'average_ear': np.mean(eye_aspects) if eye_aspects else 0.0,
            'min_ear': min(eye_aspects) if eye_aspects else 0.0,
            'max_ear': max(eye_aspects) if eye_aspects else 0.0
        }
    
    def _estimate_ear(self, gradient, intensity, height) -> float:
        """Estimate eye aspect ratio from simple features"""
        # Normalize features
        gradient_score = min(gradient / 50.0, 1.0)
        intensity_score = min(intensity / 150.0, 1.0)
        height_score = min(height / 30.0, 1.0)
        
        # Combine scores (higher = more open)
        ear = (gradient_score * 0.5 + intensity_score * 0.3 + height_score * 0.2)
        return ear
    
    def _determine_eye_state(self, metrics: Dict[str, float]) -> Tuple[str, float, bool]:
        """Determine eye state from metrics"""
        ear = metrics['average_ear']
        
        if ear < self.EAR_THRESHOLD_CLOSED:
            return 'closed', 0.8, False
        elif ear < self.EAR_THRESHOLD_HALF:
            return 'half_closed', 0.7, False
        elif ear < self.EAR_THRESHOLD_OPEN:
            return 'partially_open', 0.6, True  # Acceptable but not ideal
        else:
            return 'open', 0.9, True
    
    def select_best_frame(self, frame_paths: List[str], target_emotion: str = None) -> str:
        """
        Select the best frame from a list, avoiding half-closed eyes
        
        Args:
            frame_paths: List of frame file paths
            target_emotion: Optional emotion to match
            
        Returns:
            Path to the best frame
        """
        frame_scores = []
        
        for frame_path in frame_paths:
            eye_state = self.check_eyes_state(frame_path)
            
            # Calculate score
            score = 0.0
            
            # Eye state scoring
            if eye_state['state'] == 'open':
                score += 1.0
            elif eye_state['state'] == 'partially_open':
                score += 0.7
            elif eye_state['state'] == 'half_closed':
                score += 0.2
            else:
                score += 0.1
            
            # Confidence bonus
            score += eye_state['confidence'] * 0.3
            
            # Suitability check
            if not eye_state['suitable_for_comic']:
                score *= 0.5  # Penalize unsuitable frames
            
            frame_scores.append((frame_path, score, eye_state))
        
        # Sort by score and return best
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        
        if frame_scores:
            best_frame, best_score, best_state = frame_scores[0]
            print(f"  👁️ Selected frame with {best_state['state']} eyes (score: {best_score:.2f})")
            return best_frame
        
        return frame_paths[0] if frame_paths else None


def enhance_frame_selection(video_path: str, subtitle, output_dir: str, frames_to_extract: int = 5):
    """
    Extract multiple frames and select the best one (no half-closed eyes)
    
    Args:
        video_path: Path to video file
        subtitle: Subtitle object with start/end times
        output_dir: Directory to save the selected frame
        frames_to_extract: Number of candidate frames to extract
        
    Returns:
        Path to the selected frame
    """
    import tempfile
    
    detector = EyeStateDetector()
    
    # Create temp directory for candidate frames
    temp_dir = tempfile.mkdtemp()
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate time range
        start_time = subtitle.start.total_seconds()
        end_time = subtitle.end.total_seconds()
        duration = end_time - start_time
        
        # Extract multiple frames across the subtitle duration
        candidate_frames = []
        
        for i in range(frames_to_extract):
            # Distribute frames evenly across the duration
            time_offset = (i + 1) / (frames_to_extract + 1) * duration
            timestamp = start_time + time_offset
            frame_num = int(timestamp * fps)
            
            # Extract frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                temp_path = os.path.join(temp_dir, f"candidate_{i}.png")
                cv2.imwrite(temp_path, frame)
                candidate_frames.append(temp_path)
        
        cap.release()
        
        # Select best frame
        if candidate_frames:
            best_frame_path = detector.select_best_frame(candidate_frames)
            
            # Copy best frame to output
            if best_frame_path:
                output_path = os.path.join(output_dir, f"frame_{subtitle.index:03d}.png")
                img = cv2.imread(best_frame_path)
                cv2.imwrite(output_path, img)
                return output_path
        
    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return None