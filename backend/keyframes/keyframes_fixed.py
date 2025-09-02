"""
Fixed keyframe generation that ensures 48 frames are properly extracted
"""

import os
import cv2
import srt
from typing import List
from backend.utils import copy_and_rename_file

def generate_keyframes_fixed(video_path: str, story_subs: List, max_frames: int = 48):
    """
    Generate keyframes based on story moments - FIXED VERSION
    
    Args:
        video_path: Path to video file
        story_subs: List of subtitle objects for key story moments
        max_frames: Maximum number of frames to extract (default 48)
    """
    
    print(f"🎯 Generating {len(story_subs)} keyframes (target: {max_frames})")
    
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
    
    # Extract frames
    extracted_count = 0
    
    for i, sub in enumerate(story_subs[:max_frames]):
        try:
            # Calculate frame position (middle of subtitle duration)
            timestamp = (sub.start.total_seconds() + sub.end.total_seconds()) / 2
            frame_num = int(timestamp * fps)
            
            # Ensure frame number is valid
            frame_num = min(frame_num, total_frames - 1)
            
            # Extract frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                output_path = os.path.join(final_dir, f"frame{extracted_count:03d}.png")
                cv2.imwrite(output_path, frame)
                extracted_count += 1
                
                if i % 10 == 0 or i == len(story_subs) - 1:
                    print(f"✅ Extracted frame {i+1}/{len(story_subs)}: {sub.content[:40]}...")
            else:
                print(f"⚠️ Failed to extract frame for segment {i+1}")
                
        except Exception as e:
            print(f"❌ Error processing segment {i+1}: {e}")
    
    cap.release()
    
    # If we didn't get enough frames, extract more evenly
    if extracted_count < max_frames and extracted_count < 10:
        print(f"⚠️ Only extracted {extracted_count} frames, extracting more...")
        _extract_evenly_distributed_frames(video_path, final_dir, extracted_count, max_frames)
    
    # Final count
    final_frames = len([f for f in os.listdir(final_dir) if f.endswith('.png')])
    print(f"✅ Total frames in {final_dir}: {final_frames}")
    
    return final_frames > 0

def _extract_evenly_distributed_frames(video_path: str, output_dir: str, start_count: int, target_count: int):
    """Extract frames evenly distributed across the video"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    needed = target_count - start_count
    step = total_frames / needed if needed > 0 else 1
    
    count = start_count
    for i in range(needed):
        frame_num = int(i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            output_path = os.path.join(output_dir, f"frame{count:03d}.png")
            cv2.imwrite(output_path, frame)
            count += 1
    
    cap.release()
    print(f"✅ Extracted {count - start_count} additional frames")