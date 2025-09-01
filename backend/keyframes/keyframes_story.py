"""
Story-based Keyframe Extraction
Generates keyframes based on meaningful story moments
"""

import os
import srt
import cv2
import json
import numpy as np
from typing import List, Dict
from backend.keyframes.extract_frames import extract_frames
from backend.utils import copy_and_rename_file

def generate_keyframes_story(video_path: str, filtered_subtitles: List = None, max_frames: int = 12):
    """Generate keyframes based on story moments
    
    Args:
        video_path: Path to video file
        filtered_subtitles: List of filtered subtitle objects (if provided)
        max_frames: Maximum number of frames to generate
    """
    print("📖 Generating story-based keyframes...")
    
    # If filtered subtitles provided, use them
    if filtered_subtitles:
        subs = filtered_subtitles
        print(f"Using {len(subs)} pre-filtered story moments")
    else:
        # Read subtitle file
        try:
            with open("test1.srt") as f:
                data = f.read()
            all_subs = list(srt.parse(data))
            
            # Limit to reasonable number
            if len(all_subs) > max_frames:
                # Take evenly distributed samples
                step = len(all_subs) // max_frames
                subs = all_subs[::step][:max_frames]
                print(f"Sampled {len(subs)} from {len(all_subs)} subtitles")
            else:
                subs = all_subs
                
        except Exception as e:
            print(f"❌ Error reading subtitles: {e}")
            return False
    
    # Create final directory
    final_dir = os.path.join("frames", "final")
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    
    # Clear existing frames
    for f in os.listdir(final_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(final_dir, f))
    
    frame_counter = 0
    total_subs = len(subs)
    
    print(f"🎯 Processing {total_subs} story segments...")
    
    # Process each subtitle segment
    for i, sub in enumerate(subs):
        print(f"📝 Segment {i+1}/{total_subs}: {sub.content[:50]}...")
        
        # Create segment directory
        sub_dir = f"frames/sub{sub.index}"
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        
        try:
            # Extract 1-2 frames per segment for better coverage
            frames_per_segment = 1 if total_subs > 10 else 2
            
            frames = extract_frames(
                video_path, 
                sub_dir, 
                sub.start.total_seconds(),
                sub.end.total_seconds(),
                frames_per_segment
            )
            
            if frames:
                # Select best frame (middle one if multiple)
                best_frame_idx = len(frames) // 2
                best_frame = frames[best_frame_idx]
                
                # Copy to final directory
                src = os.path.join(sub_dir, best_frame)
                dst_filename = f"frame{frame_counter:03d}.png"
                
                copy_and_rename_file(src, final_dir, dst_filename)
                frame_counter += 1
                
                print(f"✅ Selected frame for segment {i+1}")
            else:
                print(f"⚠️ No frames extracted for segment {i+1}")
                
        except Exception as e:
            print(f"❌ Error processing segment {i+1}: {e}")
            continue
    
    # Verify we have enough frames
    if frame_counter < 5:
        print(f"⚠️ Only {frame_counter} frames generated, trying to extract more...")
        # Extract additional frames from video directly
        _extract_backup_frames(video_path, final_dir, frame_counter, min(10, max_frames))
    
    print(f"✅ Generated {frame_counter} keyframes")
    
    # Save frame metadata
    _save_frame_metadata(final_dir, subs[:frame_counter])
    
    return True

def _extract_backup_frames(video_path: str, output_dir: str, start_idx: int, target_count: int):
    """Extract backup frames if not enough story frames"""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        # Extract evenly spaced frames
        interval = duration / (target_count - start_idx)
        
        for i in range(start_idx, target_count):
            timestamp = i * interval
            frame_num = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                output_path = os.path.join(output_dir, f"frame{i:03d}.png")
                cv2.imwrite(output_path, frame)
                print(f"✅ Extracted backup frame {i}")
        
        cap.release()
        
    except Exception as e:
        print(f"❌ Backup frame extraction failed: {e}")

def _save_frame_metadata(output_dir: str, subtitles: List):
    """Save metadata about which frames correspond to which subtitles"""
    metadata = []
    
    for i, sub in enumerate(subtitles):
        metadata.append({
            'frame': f'frame{i:03d}.png',
            'subtitle': sub.content,
            'start': str(sub.start),
            'end': str(sub.end),
            'index': sub.index
        })
    
    metadata_path = os.path.join(output_dir, 'frame_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved frame metadata to {metadata_path}")