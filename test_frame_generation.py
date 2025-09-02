#!/usr/bin/env python3
"""
Test frame generation to diagnose issues
"""

import os
import sys
import cv2

sys.path.insert(0, '/workspace')

def test_frame_extraction():
    """Test if we can extract frames from video"""
    
    print("🧪 Testing Frame Extraction")
    print("=" * 50)
    
    # Check video
    video_path = 'video/uploaded.mp4'
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return False
    
    # Test video reading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video file")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"✅ Video loaded successfully")
    print(f"   FPS: {fps}")
    print(f"   Frames: {frame_count}")
    print(f"   Duration: {duration:.2f} seconds")
    
    # Test frame extraction
    test_dir = 'frames/test'
    os.makedirs(test_dir, exist_ok=True)
    
    # Extract a test frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
    ret, frame = cap.read()
    
    if ret:
        test_path = os.path.join(test_dir, 'test_frame.png')
        cv2.imwrite(test_path, frame)
        print(f"✅ Test frame extracted to: {test_path}")
    else:
        print("❌ Failed to extract test frame")
    
    cap.release()
    return True

def check_directories():
    """Check all relevant directories"""
    
    print("\n📁 Directory Status")
    print("=" * 50)
    
    dirs = {
        'video': 'Video files',
        'frames': 'Frame extraction root',
        'frames/final': 'Final frames for comic',
        'frames/cropped': 'Cropped frames',
        'output': 'Comic output',
        'audio': 'Audio/subtitle files'
    }
    
    for dir_path, desc in dirs.items():
        exists = os.path.exists(dir_path)
        status = "✅" if exists else "❌"
        print(f"{status} {dir_path}: {desc}")
        
        if exists and dir_path == 'frames/final':
            files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
            print(f"   Contains {len(files)} PNG files")

def simulate_48_frame_extraction():
    """Simulate proper 48 frame extraction"""
    
    print("\n🔧 Simulating Proper Frame Extraction")
    print("=" * 50)
    
    video_path = 'video/uploaded.mp4'
    if not os.path.exists(video_path):
        print("❌ No video to process")
        return
    
    # Ensure output directory
    final_dir = 'frames/final'
    os.makedirs(final_dir, exist_ok=True)
    
    # Clear existing frames
    for f in os.listdir(final_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(final_dir, f))
    
    # Extract 48 evenly distributed frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    target_frames = 48
    step = total_frames / target_frames
    
    extracted = 0
    for i in range(target_frames):
        frame_num = int(i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            output_path = os.path.join(final_dir, f'frame{i:03d}.png')
            cv2.imwrite(output_path, frame)
            extracted += 1
            if i % 10 == 0:
                print(f"   Extracted frame {i+1}/{target_frames}")
    
    cap.release()
    print(f"✅ Extracted {extracted} frames to {final_dir}")

if __name__ == "__main__":
    test_frame_extraction()
    check_directories()
    
    # Ask if we should extract frames
    print("\n❓ Should I extract 48 frames for testing? (This will clear existing frames)")
    # For automated testing, just do it
    simulate_48_frame_extraction()
    
    # Final check
    check_directories()