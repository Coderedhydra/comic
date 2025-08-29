"""
Simplified Keyframe Extraction
Avoids infinite loops by using basic frame selection
"""

import os
import srt
import cv2
import numpy as np
from backend.keyframes.extract_frames import extract_frames
from backend.utils import copy_and_rename_file

def generate_keyframes_simple(video):
    """Generate keyframes using simplified method"""
    print("🎯 Using simplified keyframe generation...")
    
    # Read subtitle file
    try:
        with open("test1.srt") as f:
            data = f.read()
        subs = list(srt.parse(data))
    except:
        print("❌ Error reading subtitles")
        return False
    
    # Create final directory
    final_dir = os.path.join("frames", "final")
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
        print(f"Created directory: {final_dir}")
    
    frame_counter = 1
    total_subs = len(subs)
    
    print(f"🎯 Processing {total_subs} subtitle segments...")
    
    # Process segments with simplified logic
    segments_to_process = min(16, total_subs)  # Max 16 segments
    
    for i, sub in enumerate(subs[:segments_to_process], 1):
        print(f"📝 Processing segment {i}/{segments_to_process}: {sub.content[:50]}...")
        
        # Create segment directory
        sub_dir = f"frames/sub{sub.index}"
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        
        try:
            # Extract 3-5 frames per segment (reduced from 10)
            frames = extract_frames(video, sub_dir, 
                                  sub.start.total_seconds(), 
                                  sub.end.total_seconds(), 
                                  3)  # Only 3 frames per segment
            
            if frames:
                # Simple selection: pick middle frame or best quality frame
                best_frame = _select_best_frame_simple(frames)
                
                if best_frame and frame_counter <= 16:
                    # Copy to final directory
                    final_name = f"frame{frame_counter:03}.png"
                    copy_and_rename_file(best_frame, final_dir, final_name)
                    print(f"📖 Frame {frame_counter}: {sub.content[:30]}...")
                    frame_counter += 1
                    
        except Exception as e:
            print(f"⚠️ Error processing segment {i}: {e}")
            continue
    
    frames_generated = frame_counter - 1
    print(f"✅ Generated {frames_generated} frames using simplified method")
    
    # If we don't have enough frames, duplicate some to reach 16
    if frames_generated < 16:
        print(f"🔄 Duplicating frames to reach 16 total...")
        for i in range(frames_generated + 1, 17):
            # Duplicate existing frames
            source_frame = f"frame{((i-1) % frames_generated) + 1:03}.png"
            source_path = os.path.join(final_dir, source_frame)
            target_path = os.path.join(final_dir, f"frame{i:03}.png")
            
            if os.path.exists(source_path):
                import shutil
                shutil.copy2(source_path, target_path)
                print(f"📋 Duplicated frame{i:03}.png")
    
    return True

def _select_best_frame_simple(frames):
    """Select best frame using simple criteria"""
    if not frames:
        return None
    
    if len(frames) == 1:
        return frames[0]
    
    # Simple heuristic: pick frame with most color variance (usually more interesting)
    best_frame = None
    best_score = 0
    
    for frame_path in frames:
        try:
            img = cv2.imread(frame_path)
            if img is not None:
                # Calculate color variance as a simple quality metric
                variance = np.var(img)
                if variance > best_score:
                    best_score = variance
                    best_frame = frame_path
        except:
            continue
    
    # Fallback to middle frame if variance method fails
    return best_frame if best_frame else frames[len(frames)//2]

if __name__ == "__main__":
    # Test the simplified method
    generate_keyframes_simple("video/IronMan.mp4")