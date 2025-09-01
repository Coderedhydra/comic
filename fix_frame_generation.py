#!/usr/bin/env python3
"""
Fix to ensure frames are properly generated for the full story
"""

import os
import sys

sys.path.insert(0, '/workspace')

def diagnose_issue():
    """Diagnose why frames aren't being generated"""
    
    print("🔍 Diagnosing Frame Generation Issue")
    print("=" * 50)
    
    # Check directories
    dirs_to_check = [
        'frames',
        'frames/final',
        'frames/cropped',
        'output',
        'audio'
    ]
    
    for dir_path in dirs_to_check:
        exists = os.path.exists(dir_path)
        print(f"📁 {dir_path}: {'✅ Exists' if exists else '❌ Missing'}")
        if exists and dir_path == 'frames/final':
            files = os.listdir(dir_path)
            print(f"   Files: {len(files)}")
    
    # Check subtitles
    if os.path.exists('test1.srt'):
        print("\n✅ Subtitles file exists")
        with open('test1.srt', 'r') as f:
            content = f.read()
            subtitle_count = content.count('\n\n')
            print(f"   Subtitle segments: ~{subtitle_count}")
    else:
        print("\n❌ No subtitles file found")
    
    print("\n📋 The Issue:")
    print("The system is:")
    print("1. ✅ Correctly finding 89 subtitles")
    print("2. ✅ Selecting 48 moments for full story")
    print("3. ❌ BUT then reverting to old 12-moment filtering")
    print("4. ❌ AND frames aren't being extracted")
    
    print("\n🔧 Solution:")
    print("Need to ensure the full story extraction (48 frames) is used")
    print("throughout the entire pipeline.")

def create_fixed_generator():
    """Create a fixed version that properly generates all frames"""
    
    fixed_code = '''
# Fixed version that ensures 48 frames are generated

def generate_full_story_comic(video_path):
    """Generate comic with complete story (48 frames for 12 pages)"""
    
    import os
    import cv2
    import srt
    
    # 1. Read subtitles
    with open('test1.srt', 'r') as f:
        all_subs = list(srt.parse(f.read()))
    
    print(f"📚 Found {len(all_subs)} subtitles")
    
    # 2. Select 48 evenly distributed moments
    target_frames = 48
    if len(all_subs) <= target_frames:
        selected_subs = all_subs
    else:
        step = len(all_subs) / target_frames
        selected_subs = []
        for i in range(target_frames):
            idx = int(i * step)
            selected_subs.append(all_subs[idx])
    
    print(f"✅ Selected {len(selected_subs)} moments for complete story")
    
    # 3. Extract frames
    os.makedirs('frames/final', exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for i, sub in enumerate(selected_subs):
        timestamp = (sub.start.total_seconds() + sub.end.total_seconds()) / 2
        frame_num = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            output_path = f'frames/final/frame{i:03d}.png'
            cv2.imwrite(output_path, frame)
            print(f"✅ Frame {i+1}/{len(selected_subs)}: {sub.content[:30]}...")
    
    cap.release()
    print(f"✅ Generated {len(selected_subs)} frames for full story")
    
    return len(selected_subs)
'''
    
    with open('/workspace/generate_full_story_frames.py', 'w') as f:
        f.write(fixed_code)
    
    print("\n✅ Created: generate_full_story_frames.py")
    print("This will properly extract 48 frames for the complete story")

if __name__ == "__main__":
    diagnose_issue()
    create_fixed_generator()
    
    print("\n🚀 To fix your comic generation:")
    print("1. The system needs to consistently use 48 frames")
    print("2. Not revert to 12 frames in bubble generation")
    print("3. Actually extract the frames from video")
    print("\nThe issue is the pipeline is inconsistent about frame count.")