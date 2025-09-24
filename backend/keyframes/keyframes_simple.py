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
    
    # Process ALL segments to get more frames for better sync
    target_frames = min(max(48, total_subs // 4), 100)  # At least 48 frames, up to 100
    segments_to_process = min(target_frames, total_subs)
    
    print(f"🎯 Target: {target_frames} frames from {segments_to_process} subtitle segments")
    
    for i, sub in enumerate(subs[:segments_to_process], 1):
        if i % 10 == 0:  # Progress every 10 segments
            print(f"📝 Processing segment {i}/{segments_to_process}...")
        
        # Create segment directory
        sub_dir = f"frames/sub{sub.index}"
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        
        try:
            # Extract 2 frames per segment for efficiency
            frames = extract_frames(video, sub_dir, 
                                  sub.start.total_seconds(), 
                                  sub.end.total_seconds(), 
                                  2)  # 2 frames per segment for speed
            
            if frames:
                # Simple selection: pick best quality frame
                best_frame = _select_best_frame_simple(frames)
                
                if best_frame and frame_counter <= target_frames:
                    # Copy to final directory
                    final_name = f"frame{frame_counter:03}.png"
                    copy_and_rename_file(best_frame, final_dir, final_name)
                    if i <= 10:  # Show first 10 for feedback
                        print(f"📖 Frame {frame_counter}: {sub.content[:30]}...")
                    frame_counter += 1
                    
        except Exception as e:
            if i <= 5:  # Only show first few errors
                print(f"⚠️ Error processing segment {i}: {e}")
            continue
    
    frames_generated = frame_counter - 1
    print(f"✅ Generated {frames_generated} frames using enhanced method")
    
    # If we still don't have enough frames, duplicate strategically
    if frames_generated < target_frames:
        print(f"🔄 Adding more frames to reach {target_frames} total...")
        for i in range(frames_generated + 1, target_frames + 1):
            # Duplicate existing frames in a smart pattern
            source_idx = ((i-1) % frames_generated) + 1
            source_frame = f"frame{source_idx:03}.png"
            source_path = os.path.join(final_dir, source_frame)
            target_path = os.path.join(final_dir, f"frame{i:03}.png")
            
            if os.path.exists(source_path):
                import shutil
                shutil.copy2(source_path, target_path)
    
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