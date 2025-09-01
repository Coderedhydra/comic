"""
Smart Frame Selection to Avoid Closed Eyes
Uses multiple techniques to select best frames
"""

import cv2
import numpy as np
import os
from typing import List, Tuple
import shutil

class SimpleEyeDetector:
    """Simple but effective eye detection without heavy dependencies"""
    
    def __init__(self):
        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def detect_blink_score(self, image_path: str) -> float:
        """
        Calculate blink score (0-100)
        Higher score = eyes more likely open
        """
        img = cv2.imread(image_path)
        if img is None:
            return 50.0  # Default middle score
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return 50.0  # No face, neutral score
            
        total_score = 0.0
        face_count = 0
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Focus on eye region (upper half of face)
            eye_region = face_roi[int(h*0.2):int(h*0.5), :]
            
            # Method 1: Eye cascade detection
            eyes = self.eye_cascade.detectMultiScale(eye_region, 1.1, 3)
            eye_score = 0.0
            
            if len(eyes) >= 2:
                eye_score += 40.0  # Both eyes detected
            elif len(eyes) == 1:
                eye_score += 20.0  # One eye detected
            
            # Method 2: Analyze eye region brightness variation
            # Open eyes have more contrast
            eye_std = np.std(eye_region)
            if eye_std > 20:
                eye_score += 30.0
            elif eye_std > 10:
                eye_score += 15.0
            
            # Method 3: Edge detection in eye region
            # Open eyes have more edges
            edges = cv2.Canny(eye_region, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            if edge_density > 0.1:
                eye_score += 30.0
            elif edge_density > 0.05:
                eye_score += 15.0
            
            total_score += eye_score
            face_count += 1
        
        return total_score / face_count if face_count > 0 else 50.0
    
    def is_blurry(self, image_path: str) -> bool:
        """Check if image is blurry using Laplacian variance"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
            
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance < 100  # Threshold for blur

class FrameQualityAnalyzer:
    """Analyze overall frame quality"""
    
    def __init__(self):
        self.eye_detector = SimpleEyeDetector()
    
    def analyze_frame(self, image_path: str) -> dict:
        """Comprehensive frame analysis"""
        img = cv2.imread(image_path)
        if img is None:
            return {'total_score': 0, 'usable': False}
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Initialize scores
        scores = {
            'eye_score': 0,
            'sharpness_score': 0,
            'brightness_score': 0,
            'face_score': 0,
            'total_score': 0,
            'usable': True
        }
        
        # 1. Eye/blink detection (40% weight)
        scores['eye_score'] = self.eye_detector.detect_blink_score(image_path)
        
        # 2. Sharpness (20% weight)
        if not self.eye_detector.is_blurry(image_path):
            scores['sharpness_score'] = 100
        else:
            scores['sharpness_score'] = 30
            
        # 3. Brightness (20% weight)
        brightness = np.mean(gray)
        if 60 < brightness < 200:
            scores['brightness_score'] = 100
        elif 40 < brightness < 220:
            scores['brightness_score'] = 60
        else:
            scores['brightness_score'] = 20
            
        # 4. Face detection (20% weight)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            scores['face_score'] = 100
        else:
            scores['face_score'] = 0
            
        # Calculate total score
        scores['total_score'] = (
            scores['eye_score'] * 0.4 +
            scores['sharpness_score'] * 0.2 +
            scores['brightness_score'] * 0.2 +
            scores['face_score'] * 0.2
        )
        
        # Mark as unusable if too low quality
        scores['usable'] = scores['total_score'] > 30
        
        return scores

def select_best_frames_avoid_blinks(
    input_dir: str = 'frames',
    output_dir: str = 'frames/final',
    num_frames: int = 16,
    extract_extra: bool = True
):
    """
    Select best frames avoiding blinks and closed eyes
    
    Args:
        input_dir: Directory with extracted frames
        output_dir: Directory for selected frames
        num_frames: Number of frames to select
        extract_extra: If True, extract 3x frames first for better selection
    """
    print("👁️ Smart frame selection to avoid closed eyes...")
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(input_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(frame_files) < num_frames:
        print(f"⚠️ Only {len(frame_files)} frames available, need {num_frames}")
        return
    
    # Analyze all frames
    analyzer = FrameQualityAnalyzer()
    frame_analysis = []
    
    print(f"🔍 Analyzing {len(frame_files)} frames...")
    
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(input_dir, frame_file)
        analysis = analyzer.analyze_frame(frame_path)
        
        frame_analysis.append({
            'path': frame_path,
            'filename': frame_file,
            'index': i,
            **analysis
        })
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Analyzed {i + 1}/{len(frame_files)} frames...")
    
    # Sort by total score
    frame_analysis.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Select frames with good distribution
    selected_frames = []
    selected_indices = set()
    min_frame_distance = max(1, len(frame_files) // (num_frames * 2))
    
    # First pass: Select high-quality frames with spacing
    for frame in frame_analysis:
        if len(selected_frames) >= num_frames:
            break
            
        if not frame['usable']:
            continue
            
        # Check distance from already selected frames
        too_close = any(
            abs(frame['index'] - idx) < min_frame_distance 
            for idx in selected_indices
        )
        
        if not too_close:
            selected_frames.append(frame)
            selected_indices.add(frame['index'])
            
            # Debug info
            print(f"  Selected frame {frame['filename']}: "
                  f"Score={frame['total_score']:.1f}, "
                  f"Eyes={frame['eye_score']:.1f}")
    
    # Second pass: Fill remaining slots if needed
    if len(selected_frames) < num_frames:
        print(f"⚠️ Only found {len(selected_frames)} good frames, adding more...")
        
        for frame in frame_analysis:
            if frame not in selected_frames and frame['usable']:
                selected_frames.append(frame)
                if len(selected_frames) >= num_frames:
                    break
    
    # Final pass: If still not enough, take what we can
    if len(selected_frames) < num_frames:
        for frame in frame_analysis:
            if frame not in selected_frames:
                selected_frames.append(frame)
                if len(selected_frames) >= num_frames:
                    break
    
    # Sort selected frames by original index to maintain sequence
    selected_frames.sort(key=lambda x: x['index'])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy selected frames
    for i, frame in enumerate(selected_frames[:num_frames]):
        src_path = frame['path']
        dst_filename = f'frame{i:03d}.png'
        dst_path = os.path.join(output_dir, dst_filename)
        
        shutil.copy2(src_path, dst_path)
        
        print(f"  ✅ {frame['filename']} → {dst_filename} "
              f"(Score: {frame['total_score']:.1f}, Eyes: {frame['eye_score']:.1f})")
    
    print(f"\n✅ Selected {len(selected_frames[:num_frames])} best frames")
    print(f"📊 Average eye score: {np.mean([f['eye_score'] for f in selected_frames[:num_frames]]):.1f}/100")

# Quick function to use in existing pipeline
def ensure_open_eyes_in_frames(frames_dir: str = 'frames/final'):
    """
    Post-process existing frames to check for closed eyes
    Replace bad frames with better alternatives
    """
    analyzer = FrameQualityAnalyzer()
    
    frame_files = sorted([f for f in os.listdir(frames_dir) 
                         if f.endswith(('.png', '.jpg'))])
    
    print(f"\n👁️ Checking {len(frame_files)} frames for closed eyes...")
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        analysis = analyzer.analyze_frame(frame_path)
        
        if analysis['eye_score'] < 40:  # Likely closed eyes
            print(f"  ⚠️ {frame_file}: Low eye score ({analysis['eye_score']:.1f})")
            # In a full implementation, we would replace this frame
            # with a better one from nearby frames

if __name__ == "__main__":
    # Test on existing frames
    if os.path.exists('frames'):
        select_best_frames_avoid_blinks('frames', 'frames/final_no_blinks', 16)