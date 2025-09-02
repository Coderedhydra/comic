"""
Select the most engaging frames for comic generation
Focuses on visual quality and storytelling, not showing emotion labels
"""

import os
import cv2
import srt
from typing import List, Dict, Tuple
import numpy as np
from backend.enhanced_emotion_matcher import EnhancedEmotionMatcher
from backend.eye_state_detector import EyeStateDetector
from backend.emotion_aware_comic import FacialExpressionAnalyzer

def generate_keyframes_engaging(video_path: str, story_subs: List, max_frames: int = 48):
    """
    Select the most engaging frames for comic generation
    
    Criteria:
    1. Facial expression matches dialogue mood
    2. Eyes are open (no blinking)
    3. Good composition (face visible, not blurry)
    4. Dramatic/interesting moments
    """
    
    print(f"🎬 Selecting most engaging frames for comic generation...")
    print(f"📊 Processing {len(story_subs)} story moments")
    
    # Initialize analyzers (used internally, not shown to user)
    emotion_matcher = EnhancedEmotionMatcher()
    face_analyzer = FacialExpressionAnalyzer()
    eye_detector = EyeStateDetector()
    
    # Ensure output directory exists
    final_dir = "frames/final"
    os.makedirs(final_dir, exist_ok=True)
    
    # Clear existing frames
    for f in os.listdir(final_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(final_dir, f))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Analyzing video: {fps:.1f} fps, {total_frames} frames")
    print(f"🔍 Finding best frames for each story moment...")
    
    # Process each subtitle
    selected_count = 0
    
    for idx, sub in enumerate(story_subs[:max_frames]):
        # Don't show emotion analysis to user, just use it internally
        text_emotions = emotion_matcher.analyze_text_emotion(sub.content)
        target_mood = max(text_emotions.items(), 
                         key=lambda x: x[1] if x[0] != 'intensity' else 0)[0]
        
        # Progress indicator (simple, not technical)
        if idx % 5 == 0:
            print(f"  Processing moments {idx+1}-{min(idx+5, len(story_subs))}...")
        
        # Find the most engaging frame for this moment
        best_frame = find_most_engaging_frame(
            cap, sub, fps, 
            face_analyzer, eye_detector,
            target_mood, text_emotions
        )
        
        if best_frame is not None:
            # Save the selected frame
            output_path = os.path.join(final_dir, f"frame{selected_count:03d}.png")
            
            # Apply any visual enhancements for comic style
            enhanced_frame = enhance_for_comic(best_frame['image'])
            cv2.imwrite(output_path, enhanced_frame)
            
            selected_count += 1
        else:
            # Fallback: get a decent frame from the middle
            fallback_frame = get_decent_frame(cap, sub, fps)
            if fallback_frame is not None:
                output_path = os.path.join(final_dir, f"frame{selected_count:03d}.png")
                enhanced_frame = enhance_for_comic(fallback_frame)
                cv2.imwrite(output_path, enhanced_frame)
                selected_count += 1
    
    cap.release()
    
    print(f"\n✅ Selected {selected_count} engaging frames for comic")
    print(f"📁 Frames saved to: {final_dir}")
    
    return selected_count > 0


def find_most_engaging_frame(cap, subtitle, fps, face_analyzer, eye_detector, 
                            target_mood, text_emotions):
    """
    Find the most visually engaging frame for this subtitle
    
    Scoring based on:
    - Expression matching dialogue (internal, not shown)
    - Eye quality (open, alert)
    - Visual composition
    - Sharpness/clarity
    """
    
    # Time window to search
    start_time = subtitle.start.total_seconds()
    end_time = subtitle.end.total_seconds()
    duration = end_time - start_time
    
    # Extend search window slightly for better options
    search_start = max(0, start_time - 0.5)
    search_end = end_time + 0.5
    
    start_frame = int(search_start * fps)
    end_frame = int(search_end * fps)
    
    # Sample frames intelligently
    num_samples = min(15, end_frame - start_frame)
    if num_samples <= 0:
        num_samples = 5
    
    frame_step = max(1, (end_frame - start_frame) // num_samples)
    
    best_frame = None
    best_score = -1
    
    for frame_num in range(start_frame, end_frame, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue
        
        # Calculate engagement score
        score = calculate_engagement_score(
            frame, face_analyzer, eye_detector, 
            target_mood, text_emotions
        )
        
        if score > best_score:
            best_score = score
            best_frame = {
                'image': frame.copy(),
                'score': score,
                'frame_num': frame_num
            }
    
    return best_frame


def calculate_engagement_score(frame, face_analyzer, eye_detector, 
                              target_mood, text_emotions):
    """
    Calculate how engaging/suitable this frame is for the comic
    
    High scores for:
    - Good facial expressions
    - Open eyes
    - Clear image
    - Good composition
    """
    
    score = 0.0
    
    # Save temp for analysis
    temp_path = "temp_frame_analysis.png"
    cv2.imwrite(temp_path, frame)
    
    try:
        # 1. Eye quality (most important for comics)
        eye_state = eye_detector.check_eyes_state(temp_path)
        if eye_state['state'] == 'open':
            score += 3.0
        elif eye_state['state'] == 'partially_open':
            score += 1.5
        elif eye_state['state'] == 'unknown':
            score += 1.0  # No face, might be okay
        else:  # closed or half_closed
            score += 0.0  # Strong penalty
        
        # 2. Expression quality (internal matching)
        face_emotions = face_analyzer.analyze_expression(temp_path)
        
        # Check if expression matches mood
        if target_mood in face_emotions and face_emotions[target_mood] > 0.3:
            score += 2.0 * face_emotions[target_mood]
        
        # General expressiveness (any strong emotion is interesting)
        max_emotion = max(face_emotions.values())
        if max_emotion > 0.5:
            score += 1.0
        
        # 3. Image quality
        sharpness = calculate_sharpness(frame)
        score += sharpness * 0.5
        
        # 4. Composition (face detection confidence)
        if eye_state.get('confidence', 0) > 0.7:
            score += 0.5
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return score


def calculate_sharpness(frame):
    """Calculate image sharpness using Laplacian variance"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    # Normalize to 0-1 range
    # Typical variance ranges from 0 (very blurry) to 1000+ (very sharp)
    normalized = min(variance / 500.0, 1.0)
    return normalized


def enhance_for_comic(frame):
    """Apply subtle enhancements to make frame more comic-like"""
    # Just enhance contrast slightly for better comic appearance
    # No heavy processing or style changes
    
    # Increase contrast slightly
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def get_decent_frame(cap, subtitle, fps):
    """Get a decent fallback frame"""
    # Try multiple positions to find a decent frame
    positions = [0.5, 0.3, 0.7, 0.2, 0.8]  # Middle, then alternatives
    
    duration = subtitle.end.total_seconds() - subtitle.start.total_seconds()
    
    for pos in positions:
        time_offset = subtitle.start.total_seconds() + (duration * pos)
        frame_num = int(time_offset * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret and frame is not None:
            # Quick quality check
            if calculate_sharpness(frame) > 0.3:
                return frame
    
    return None