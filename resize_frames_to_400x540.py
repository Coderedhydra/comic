#!/usr/bin/env python3
"""
Resize all extracted frames to exactly 400x540 for perfect comic layout
"""

import os
import cv2
import sys

def resize_frames_to_400x540():
    """Resize all frames in frames/final to 400x540"""
    
    frames_dir = "frames/final"
    output_dir = "frames/resized_400x540"
    
    if not os.path.exists(frames_dir):
        print(f"❌ Frames directory not found: {frames_dir}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG files
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
    
    if not frame_files:
        print(f"❌ No PNG files found in {frames_dir}")
        return False
    
    print(f"📐 Resizing {len(frame_files)} frames to 400x540...")
    
    resized_count = 0
    
    for i, frame_file in enumerate(frame_files, 1):
        input_path = os.path.join(frames_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)
        
        try:
            # Read original image
            img = cv2.imread(input_path)
            if img is None:
                print(f"  ❌ Could not read {frame_file}")
                continue
            
            # Get original dimensions
            orig_height, orig_width = img.shape[:2]
            
            # Resize to 400x540 with padding (no cropping)
            # This preserves the entire image content
            resized = cv2.resize(img, (400, 540), interpolation=cv2.INTER_LANCZOS4)
            
            # Save resized image
            cv2.imwrite(output_path, resized)
            
            # Verify the output
            saved_img = cv2.imread(output_path)
            if saved_img is not None:
                saved_height, saved_width = saved_img.shape[:2]
                file_size = os.path.getsize(output_path)
                
                print(f"  ✓ {frame_file}: {orig_width}x{orig_height} → {saved_width}x{saved_height} ({file_size/1024:.1f} KB)")
                resized_count += 1
            else:
                print(f"  ❌ Failed to save {frame_file}")
                
        except Exception as e:
            print(f"  ❌ Error processing {frame_file}: {e}")
    
    print(f"✅ Successfully resized {resized_count}/{len(frame_files)} frames to 400x540")
    print(f"📁 Resized frames saved to: {output_dir}")
    
    return resized_count > 0

def test_resized_frames():
    """Test the resized frames"""
    
    output_dir = "frames/resized_400x540"
    
    if not os.path.exists(output_dir):
        print(f"❌ Resized frames directory not found: {output_dir}")
        return
    
    frame_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    
    print(f"\n🧪 Testing {len(frame_files)} resized frames:")
    
    for i, frame_file in enumerate(frame_files[:4], 1):  # Test first 4
        frame_path = os.path.join(output_dir, frame_file)
        
        try:
            img = cv2.imread(frame_path)
            if img is not None:
                height, width = img.shape[:2]
                file_size = os.path.getsize(frame_path)
                
                print(f"  Frame {i}: {width}x{height} ({file_size/1024:.1f} KB)")
                
                # Check if dimensions are correct
                if width == 400 and height == 540:
                    print(f"    ✅ Perfect dimensions!")
                else:
                    print(f"    ❌ Wrong dimensions! Expected 400x540")
            else:
                print(f"  ❌ Could not read {frame_file}")
                
        except Exception as e:
            print(f"  ❌ Error testing {frame_file}: {e}")

if __name__ == "__main__":
    print("🎬 Frame Resizer - 400x540 for Comic Layout")
    print("=" * 50)
    
    # Resize frames
    success = resize_frames_to_400x540()
    
    if success:
        # Test the results
        test_resized_frames()
        
        print("\n🎯 Next Steps:")
        print("1. Update the HTML to use frames/resized_400x540/ instead of frames/final/")
        print("2. Each panel will be exactly 400x540 pixels")
        print("3. 4 panels = 800x1080 total page size")
        print("4. No gaps, no cropping, perfect fit!")
    else:
        print("\n❌ Resizing failed. Check the frames/final directory.")