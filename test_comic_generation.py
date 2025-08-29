#!/usr/bin/env python3
"""
Test Comic Generation
Debug script to see where the process is failing
"""

import os
import sys
import time

def test_step_by_step():
    """Test each step of comic generation"""
    print("🔍 Testing Comic Generation Step by Step")
    print("=" * 50)
    
    # Step 1: Check video file
    print("1️⃣ Checking video file...")
    video_path = 'video/IronMan.mp4'
    if os.path.exists(video_path):
        print(f"✅ Video found: {video_path}")
    else:
        print(f"❌ Video not found: {video_path}")
        return False
    
    # Step 2: Test subtitle extraction
    print("\n2️⃣ Testing subtitle extraction...")
    try:
        from backend.subtitles.subs_simple import get_subtitles
        get_subtitles(video_path)
        print("✅ Subtitles extracted")
    except Exception as e:
        print(f"❌ Subtitle extraction failed: {e}")
        return False
    
    # Step 3: Test keyframe generation
    print("\n3️⃣ Testing keyframe generation...")
    try:
        from backend.keyframes.keyframes import generate_keyframes
        generate_keyframes(video_path)
        print("✅ Keyframes generated")
    except Exception as e:
        print(f"❌ Keyframe generation failed: {e}")
        return False
    
    # Step 4: Check frames directory
    print("\n4️⃣ Checking frames directory...")
    frames_dir = 'frames/final'
    if os.path.exists(frames_dir):
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        print(f"✅ Found {len(frame_files)} frames")
        if frame_files:
            print(f"   First frame: {frame_files[0]}")
    else:
        print("❌ Frames directory not found")
        return False
    
    # Step 5: Test black bar removal
    print("\n5️⃣ Testing black bar removal...")
    try:
        from backend.keyframes.keyframes import black_bar_crop
        black_x, black_y, _, _ = black_bar_crop()
        print(f"✅ Black bars removed: ({black_x}, {black_y})")
    except Exception as e:
        print(f"❌ Black bar removal failed: {e}")
        return False
    
    # Step 6: Test image enhancement (skip if too slow)
    print("\n6️⃣ Testing image enhancement (1 frame only)...")
    try:
        from backend.ai_enhanced_core import image_processor
        test_frame = os.path.join(frames_dir, frame_files[0])
        image_processor.enhance_image_quality(test_frame)
        print("✅ Image enhancement completed")
    except Exception as e:
        print(f"❌ Image enhancement failed: {e}")
        return False
    
    # Step 7: Test comic styling (1 frame only)
    print("\n7️⃣ Testing comic styling (1 frame only)...")
    try:
        from backend.ai_enhanced_core import comic_styler
        comic_styler.apply_comic_style(test_frame, style_type="modern")
        print("✅ Comic styling completed")
    except Exception as e:
        print(f"❌ Comic styling failed: {e}")
        return False
    
    # Step 8: Test layout generation
    print("\n8️⃣ Testing layout generation...")
    try:
        from backend.ai_enhanced_core import layout_optimizer
        frame_paths = [os.path.join(frames_dir, f) for f in frame_files[:4]]
        layout = layout_optimizer.optimize_layout(frame_paths, target_layout="2x2")
        print(f"✅ Layout generated: {len(layout)} panels")
    except Exception as e:
        print(f"❌ Layout generation failed: {e}")
        return False
    
    # Step 9: Test bubble creation
    print("\n9️⃣ Testing bubble creation...")
    try:
        from backend.ai_bubble_placement import ai_bubble_placer
        panel_coords = (0, 500, 0, 400)
        position = ai_bubble_placer.place_bubble_ai(test_frame, panel_coords, (100, 100), "Test dialogue")
        print(f"✅ Bubble placement: {position}")
    except Exception as e:
        print(f"❌ Bubble creation failed: {e}")
        return False
    
    # Step 10: Test output creation
    print("\n🔟 Testing output creation...")
    try:
        os.makedirs('output', exist_ok=True)
        with open('output/test.html', 'w') as f:
            f.write('<html><body><h1>Test Comic</h1></body></html>')
        print("✅ Output directory created")
    except Exception as e:
        print(f"❌ Output creation failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Comic generation should work.")
    return True

if __name__ == "__main__":
    success = test_step_by_step()
    if success:
        print("\n✅ Ready to generate comic!")
        print("Run: python app_enhanced.py")
    else:
        print("\n❌ Some tests failed. Check the errors above.")