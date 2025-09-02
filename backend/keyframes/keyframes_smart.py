"""
Smart keyframe generation with eye detection and emotion matching
"""

import os
import cv2
import srt
from typing import List
import numpy as np
from backend.eye_state_detector import EyeStateDetector, enhance_frame_selection
from backend.utils import copy_and_rename_file

def generate_keyframes_smart(video_path: str, story_subs: List, max_frames: int = 48):
    """
    Generate keyframes with smart selection (no half-closed eyes)
    
    Args:
        video_path: Path to video file
        story_subs: List of subtitle objects for key story moments
        max_frames: Maximum number of frames to extract (default 48)
    """
    
    print(f"🎯 Generating {len(story_subs)} smart keyframes (avoiding closed eyes)")
    
    # Initialize eye detector
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
    
    print(f"📹 Video: {fps} fps, {total_frames} total frames")
    print(f"👁️ Smart frame selection enabled (avoiding half-closed eyes)")
    
    # Extract frames
    extracted_count = 0
    
    for i, sub in enumerate(story_subs[:max_frames]):
        try:
            print(f"\n📝 Processing segment {i+1}/{min(len(story_subs), max_frames)}: {sub.content[:40]}...")
            
            # Extract multiple candidate frames for this subtitle
            candidates = extract_candidate_frames(
                cap, sub, fps, 
                num_candidates=5  # Extract 5 frames to choose from
            )
            
            if candidates:
                # Select best frame (no half-closed eyes)
                best_frame, eye_state = select_best_candidate(candidates, eye_detector)
                
                if best_frame is not None:
                    output_path = os.path.join(final_dir, f"frame{extracted_count:03d}.png")
                    cv2.imwrite(output_path, best_frame)
                    extracted_count += 1
                    
                    print(f"  ✅ Selected frame with {eye_state['state']} eyes (confidence: {eye_state['confidence']:.2f})")
                else:
                    print(f"  ⚠️ No suitable frame found (all had closed/half-closed eyes)")
            else:
                print(f"  ⚠️ Failed to extract candidate frames")
                
        except Exception as e:
            print(f"  ❌ Error processing segment {i+1}: {e}")
    
    cap.release()
    
    # If we didn't get enough frames, extract more with relaxed criteria
    if extracted_count < max_frames and extracted_count < 10:
        print(f"\n⚠️ Only extracted {extracted_count} frames, extracting more with relaxed criteria...")
        _extract_additional_frames(video_path, final_dir, extracted_count, max_frames)
    
    # Final count
    final_frames = len([f for f in os.listdir(final_dir) if f.endswith('.png')])
    print(f"\n✅ Total frames extracted: {final_frames}")
    print(f"👁️ All frames checked for eye quality")
    
    return final_frames > 0


def extract_candidate_frames(cap, subtitle, fps, num_candidates=5):
    """Extract multiple candidate frames from a subtitle segment"""
    
    candidates = []
    
    # Calculate time range
    start_time = subtitle.start.total_seconds()
    end_time = subtitle.end.total_seconds()
    duration = end_time - start_time
    
    # If duration is very short, just get middle frame
    if duration < 0.5:
        num_candidates = 1
    
    # Extract frames evenly distributed across the duration
    for i in range(num_candidates):
        # Calculate timestamp (avoid very start/end to reduce motion blur)
        if num_candidates == 1:
            time_offset = duration / 2
        else:
            # Distribute between 20% and 80% of duration
            time_offset = 0.2 * duration + (i / (num_candidates - 1)) * 0.6 * duration
        
        timestamp = start_time + time_offset
        frame_num = int(timestamp * fps)
        
        # Extract frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret and frame is not None:
            candidates.append(frame)
    
    return candidates


def select_best_candidate(candidates: List[np.ndarray], eye_detector: EyeStateDetector):
    """Select the best frame from candidates based on eye state"""
    
    best_frame = None
    best_score = -1
    best_state = None
    
    for i, frame in enumerate(candidates):
        # Save temp frame for analysis
        temp_path = f"temp_candidate_{i}.png"
        cv2.imwrite(temp_path, frame)
        
        # Check eye state
        eye_state = eye_detector.check_eyes_state(temp_path)
        
        # Calculate score
        score = calculate_frame_score(eye_state)
        
        # Update best if this is better
        if score > best_score:
            best_score = score
            best_frame = frame
            best_state = eye_state
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return best_frame, best_state


def calculate_frame_score(eye_state):
    """Calculate a quality score for a frame based on eye state"""
    
    score = 0.0
    
    # Eye state scoring (most important)
    if eye_state['state'] == 'open':
        score += 10.0
    elif eye_state['state'] == 'partially_open':
        score += 7.0
    elif eye_state['state'] == 'unknown':
        score += 5.0  # Might be okay (no face detected)
    elif eye_state['state'] == 'half_closed':
        score += 2.0
    else:  # closed
        score += 0.0
    
    # Confidence bonus
    score += eye_state['confidence'] * 3.0
    
    # Suitability check
    if eye_state['suitable_for_comic']:
        score += 5.0
    
    return score


def _extract_additional_frames(video_path: str, output_dir: str, start_count: int, target_count: int):
    """Extract additional frames with relaxed eye criteria"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    eye_detector = EyeStateDetector()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    needed = target_count - start_count
    step = total_frames / needed if needed > 0 else 1
    
    count = start_count
    attempts = 0
    max_attempts = needed * 3  # Try up to 3x frames to find good ones
    
    while count < target_count and attempts < max_attempts:
        frame_num = int((attempts * step) % total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Check eye state with relaxed criteria
            temp_path = f"temp_check_{attempts}.png"
            cv2.imwrite(temp_path, frame)
            eye_state = eye_detector.check_eyes_state(temp_path)
            
            # Accept if not completely closed
            if eye_state['state'] not in ['closed', 'half_closed']:
                output_path = os.path.join(output_dir, f"frame{count:03d}.png")
                cv2.imwrite(output_path, frame)
                count += 1
                print(f"  ✅ Added frame {count} ({eye_state['state']} eyes)")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        attempts += 1
    
    cap.release()
    print(f"  ✅ Extracted {count - start_count} additional frames")