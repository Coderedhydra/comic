"""
Enhanced Keyframe Generation that Avoids Closed Eyes
"""

import os
import sys
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.keyframes.keyframes import generate_keyframes as original_generate_keyframes
from backend.keyframes.extract_frames import extract_frames
from backend.smart_frame_selector import select_best_frames_avoid_blinks

def generate_keyframes_no_blinks(video_path):
    """
    Generate keyframes while avoiding frames with closed eyes
    
    This is a drop-in replacement for generate_keyframes that:
    1. Extracts 3x more frames than needed
    2. Analyzes each frame for eye state
    3. Selects the best 16 frames with open eyes
    """
    print("🎬 Enhanced keyframe generation (avoiding closed eyes)...")
    
    # Step 1: Extract more frames than needed (48 frames for 16 final)
    print("📹 Extracting extra frames for better selection...")
    extract_frames(video_path, num_frames=48, output_dir='frames_temp')
    
    # Step 2: Analyze and select best frames
    print("👁️ Selecting frames with open eyes...")
    select_best_frames_avoid_blinks(
        input_dir='frames_temp',
        output_dir='frames',
        num_frames=16
    )
    
    # Step 3: Continue with normal keyframe processing
    print("🎯 Processing selected keyframes...")
    result = original_generate_keyframes(video_path)
    
    # Cleanup temporary frames
    if os.path.exists('frames_temp'):
        shutil.rmtree('frames_temp')
    
    return result

def quick_fix_existing_frames():
    """
    Quick fix for existing frames with closed eyes
    Can be run on already extracted frames
    """
    if not os.path.exists('frames/final'):
        print("❌ No frames found in frames/final")
        return
        
    # Create backup
    if os.path.exists('frames/final_backup'):
        shutil.rmtree('frames/final_backup')
    shutil.copytree('frames/final', 'frames/final_backup')
    
    # Re-select frames from all available
    if os.path.exists('frames'):
        print("🔄 Re-selecting frames to avoid closed eyes...")
        
        # Get all frames (not just final)
        all_frames = [f for f in os.listdir('frames') 
                     if f.startswith('frame') and f.endswith('.png')]
        
        if len(all_frames) > 16:
            # We have more frames to choose from
            select_best_frames_avoid_blinks(
                input_dir='frames',
                output_dir='frames/final_fixed',
                num_frames=16
            )
            
            # Replace final with fixed
            if os.path.exists('frames/final_fixed'):
                shutil.rmtree('frames/final')
                shutil.move('frames/final_fixed', 'frames/final')
                print("✅ Frames updated with better selections")
        else:
            print("⚠️ Not enough extra frames for re-selection")
    
    return True

# Make it easy to use
def smart_generate_keyframes(video_path):
    """Alias for generate_keyframes_no_blinks"""
    return generate_keyframes_no_blinks(video_path)

if __name__ == "__main__":
    # Test or fix existing frames
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--fix":
            quick_fix_existing_frames()
        else:
            smart_generate_keyframes(sys.argv[1])
    else:
        print("Usage:")
        print("  python keyframes_no_blinks.py video.mp4  # Generate new keyframes")
        print("  python keyframes_no_blinks.py --fix      # Fix existing frames")