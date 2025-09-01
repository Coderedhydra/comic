#!/usr/bin/env python3
"""
Fix Closed Eyes in Comic Generation
Run this before or after frame extraction
"""

import os
import sys
import cv2
import numpy as np

# Add to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.smart_frame_selector import select_best_frames_avoid_blinks, ensure_open_eyes_in_frames
from backend.keyframes.keyframes_no_blinks import quick_fix_existing_frames

def fix_closed_eyes_in_video(video_path=None):
    """Complete solution to fix closed eyes"""
    
    print("👁️ FIXING CLOSED EYES IN COMIC GENERATION")
    print("=" * 50)
    
    # Option 1: Fix existing frames if they exist
    if os.path.exists('frames/final'):
        print("\n📁 Found existing frames, analyzing...")
        ensure_open_eyes_in_frames('frames/final')
        
        response = input("\n❓ Do you want to re-select frames? (y/n): ")
        if response.lower() == 'y':
            quick_fix_existing_frames()
            print("✅ Frames have been re-selected!")
    
    # Option 2: Extract new frames with eye detection
    elif video_path and os.path.exists(video_path):
        print(f"\n🎬 Processing video: {video_path}")
        
        # Use the enhanced keyframe generation
        from backend.keyframes.keyframes_no_blinks import generate_keyframes_no_blinks
        generate_keyframes_no_blinks(video_path)
        
        print("✅ Keyframes generated with eye detection!")
    
    else:
        print("\n❌ No frames or video found")
        print("\nUsage:")
        print("  python fix_closed_eyes.py                    # Fix existing frames")
        print("  python fix_closed_eyes.py video.mp4          # Process new video")

def integrate_with_app():
    """
    Modify app_enhanced.py to use eye detection
    
    Add this to your comic generation pipeline:
    """
    code = '''
# In app_enhanced.py, replace:
# generate_keyframes(self.video_path)

# With:
from backend.keyframes.keyframes_no_blinks import generate_keyframes_no_blinks
generate_keyframes_no_blinks(self.video_path)
    '''
    
    print("\n📝 To integrate with your app, add this code:")
    print(code)
    
    # Or automatically patch it
    response = input("\n❓ Do you want to automatically patch app_enhanced.py? (y/n): ")
    if response.lower() == 'y':
        patch_app_enhanced()

def patch_app_enhanced():
    """Patch app_enhanced.py to use eye detection"""
    try:
        # Read the file
        with open('app_enhanced.py', 'r') as f:
            content = f.read()
        
        # Replace the import
        if 'from backend.keyframes.keyframes import generate_keyframes' in content:
            content = content.replace(
                'from backend.keyframes.keyframes import generate_keyframes',
                'from backend.keyframes.keyframes_no_blinks import generate_keyframes_no_blinks as generate_keyframes'
            )
            
            # Write back
            with open('app_enhanced.py', 'w') as f:
                f.write(content)
                
            print("✅ app_enhanced.py has been patched!")
            print("🎉 Your comic generation will now avoid closed eyes!")
        else:
            print("⚠️ Could not find the import to patch")
            
    except Exception as e:
        print(f"❌ Error patching file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fix_closed_eyes_in_video(sys.argv[1])
    else:
        fix_closed_eyes_in_video()
        
    # Show integration options
    integrate_with_app()