"""
Emotion-based keyframe selection - analyzes emotions FIRST, then selects matching frames
"""

import os
import cv2
import srt
from typing import List, Dict, Tuple
import numpy as np
from backend.enhanced_emotion_matcher import EnhancedEmotionMatcher
from backend.eye_state_detector import EyeStateDetector
from backend.emotion_aware_comic import FacialExpressionAnalyzer

def generate_keyframes_emotion_based(video_path: str, story_subs: List, max_frames: int = 48):
    """
    Generate keyframes by matching facial expressions to dialogue emotions
    
    This analyzes emotions FIRST, then finds the best matching frames
    """
    
    print(f"🎭 Emotion-Based Frame Selection (Analyzing emotions BEFORE frame selection)")
    print(f"📝 Analyzing {len(story_subs)} dialogues for emotions...")
    
    # Initialize analyzers
    emotion_matcher = EnhancedEmotionMatcher()
    face_analyzer = FacialExpressionAnalyzer()
    eye_detector = EyeStateDetector()
    
    # Step 1: Analyze all dialogue emotions first
    dialogue_emotions = []
    for i, sub in enumerate(story_subs[:max_frames]):
        text_emotions = emotion_matcher.analyze_text_emotion(sub.content)
        dominant_emotion = max(text_emotions.items(), 
                             key=lambda x: x[1] if x[0] != 'intensity' else 0)[0]
        
        dialogue_emotions.append({
            'subtitle': sub,
            'text': sub.content,
            'emotions': text_emotions,
            'dominant': dominant_emotion,
            'start_time': sub.start.total_seconds(),
            'end_time': sub.end.total_seconds()
        })
        
        print(f"  📖 Dialogue {i+1}: '{sub.content[:40]}...' → {dominant_emotion}")
    
    print(f"\n🎬 Scanning video for matching facial expressions...")
    
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
    
    print(f"📹 Video: {fps} fps, {total_frames} total frames")
    
    # Step 2: For each dialogue, find the best matching frame
    selected_frames = []
    
    for idx, dialogue_data in enumerate(dialogue_emotions):
        print(f"\n🔍 Finding best frame for dialogue {idx+1}: {dialogue_data['dominant']} emotion")
        
        best_frame = find_best_emotional_frame(
            cap, dialogue_data, fps, 
            face_analyzer, eye_detector,
            scan_window=2.0  # Scan 2 seconds around dialogue
        )
        
        if best_frame is not None:
            # Save the selected frame
            output_path = os.path.join(final_dir, f"frame{idx:03d}.png")
            cv2.imwrite(output_path, best_frame['image'])
            
            selected_frames.append({
                'path': output_path,
                'dialogue': dialogue_data,
                'face_emotion': best_frame['face_emotion'],
                'match_score': best_frame['match_score'],
                'eye_state': best_frame['eye_state']
            })
            
            print(f"  ✅ Selected frame with {best_frame['face_emotion']} face " +
                  f"(match: {best_frame['match_score']:.0%}, eyes: {best_frame['eye_state']})")
        else:
            print(f"  ⚠️ No good emotional match found, using default frame")
            # Fallback: just get middle frame
            fallback_frame = get_fallback_frame(cap, dialogue_data, fps)
            if fallback_frame is not None:
                output_path = os.path.join(final_dir, f"frame{idx:03d}.png")
                cv2.imwrite(output_path, fallback_frame)
                selected_frames.append({
                    'path': output_path,
                    'dialogue': dialogue_data,
                    'face_emotion': 'unknown',
                    'match_score': 0.0,
                    'eye_state': 'unknown'
                })
    
    cap.release()
    
    # Summary
    print(f"\n📊 Emotion-Based Selection Summary:")
    print(f"✅ Selected {len(selected_frames)} frames based on emotion matching")
    
    if selected_frames:
        good_matches = sum(1 for f in selected_frames if f['match_score'] > 0.7)
        print(f"😊 Good emotion matches: {good_matches}/{len(selected_frames)}")
        
        # Count emotions
        emotion_counts = {}
        for frame in selected_frames:
            emotion = frame['face_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("\n🎭 Selected facial expressions:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {count} frames")
    
    return len(selected_frames) > 0


def find_best_emotional_frame(cap, dialogue_data, fps, face_analyzer, eye_detector, scan_window=2.0):
    """
    Find the best frame that matches the dialogue emotion
    
    Scans frames around the dialogue timing to find matching facial expression
    """
    
    target_emotion = dialogue_data['dominant']
    text_emotions = dialogue_data['emotions']
    
    # Calculate scan range
    center_time = (dialogue_data['start_time'] + dialogue_data['end_time']) / 2
    start_time = max(0, center_time - scan_window)
    end_time = center_time + scan_window
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Sample frames (don't check every single frame)
    num_samples = min(20, end_frame - start_frame)  # Check up to 20 frames
    if num_samples <= 0:
        num_samples = 5
    
    frame_step = max(1, (end_frame - start_frame) // num_samples)
    
    best_match = None
    best_score = -1
    
    for frame_num in range(start_frame, end_frame, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue
        
        # Save temp frame for analysis
        temp_path = f"temp_emotion_check_{frame_num}.png"
        cv2.imwrite(temp_path, frame)
        
        try:
            # Check eye state first
            eye_state = eye_detector.check_eyes_state(temp_path)
            
            # Skip if eyes are closed or half-closed
            if eye_state['state'] in ['closed', 'half_closed']:
                continue
            
            # Analyze facial expression
            face_emotions = face_analyzer.analyze_expression(temp_path)
            face_dominant = max(face_emotions.items(), 
                              key=lambda x: x[1] if x[0] != 'intensity' else 0)[0]
            
            # Calculate match score
            score = calculate_emotion_match_score(text_emotions, face_emotions, target_emotion)
            
            # Bonus for good eye state
            if eye_state['state'] == 'open':
                score *= 1.2
            
            # Update best match
            if score > best_score:
                best_score = score
                best_match = {
                    'image': frame.copy(),
                    'face_emotion': face_dominant,
                    'face_emotions': face_emotions,
                    'match_score': min(score, 1.0),
                    'eye_state': eye_state['state'],
                    'frame_num': frame_num
                }
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return best_match


def calculate_emotion_match_score(text_emotions: Dict, face_emotions: Dict, target_emotion: str) -> float:
    """Calculate how well the face matches the text emotion"""
    
    score = 0.0
    
    # Direct match bonus
    if target_emotion in face_emotions and face_emotions[target_emotion] > 0.3:
        score += face_emotions[target_emotion] * 2.0
    
    # Check if face has the target emotion as dominant
    face_dominant = max(face_emotions.items(), 
                       key=lambda x: x[1] if x[0] != 'intensity' else 0)[0]
    if face_dominant == target_emotion:
        score += 0.5
    
    # Compare all emotions
    for emotion in ['happy', 'sad', 'angry', 'surprised', 'scared', 'neutral']:
        text_val = text_emotions.get(emotion, 0)
        face_val = face_emotions.get(emotion, 0)
        
        if text_val > 0.3 and face_val > 0.3:
            # Both have this emotion
            score += min(text_val, face_val) * 0.5
        elif text_val > 0.5 and face_val < 0.2:
            # Text has emotion but face doesn't - penalty
            score -= 0.2
    
    # Intensity matching
    text_intensity = text_emotions.get('intensity', 0.5)
    face_intensity = face_emotions.get('intensity', 0.5)
    intensity_diff = abs(text_intensity - face_intensity)
    score += (1 - intensity_diff) * 0.3
    
    return max(0, score)


def get_fallback_frame(cap, dialogue_data, fps):
    """Get a fallback frame from the middle of the dialogue"""
    
    middle_time = (dialogue_data['start_time'] + dialogue_data['end_time']) / 2
    frame_num = int(middle_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    return frame if ret else None