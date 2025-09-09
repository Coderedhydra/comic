#!/usr/bin/env python3
"""
Test script for Unity Comic Generator
Creates 48 pages with 2x2 grid layout for Unity integration
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

def create_test_frames():
    """Create test frames for Unity comic generation"""
    print("🎬 Creating test frames...")
    
    # Create test frames directory
    os.makedirs('frames/unity_extracted', exist_ok=True)
    
    # Create 192 test frames (48 pages * 4 panels)
    total_frames = 48 * 4
    
    for i in range(total_frames):
        # Create a test image with gradient background
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Create gradient background
        for y in range(720):
            for x in range(1280):
                img[y, x] = [
                    int(255 * (x / 1280)),  # Red gradient
                    int(255 * (y / 720)),   # Green gradient
                    int(255 * ((x + y) / (1280 + 720)))  # Blue gradient
                ]
        
        # Add frame number text
        cv2.putText(img, f"Frame {i+1:03d}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        
        # Add page info
        page_num = (i // 4) + 1
        panel_num = (i % 4) + 1
        cv2.putText(img, f"Page {page_num:02d} Panel {panel_num}", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Save frame
        frame_path = f'frames/unity_extracted/frame_{i:03d}.png'
        cv2.imwrite(frame_path, img)
    
    print(f"✅ Created {total_frames} test frames")
    return [f'frames/unity_extracted/frame_{i:03d}.png' for i in range(total_frames)]

def create_test_subtitles():
    """Create test subtitles"""
    print("💬 Creating test subtitles...")
    
    subtitles = []
    total_panels = 48 * 4
    
    for i in range(total_panels):
        page_num = (i // 4) + 1
        panel_num = (i % 4) + 1
        
        subtitles.append({
            'text': f"Page {page_num} Panel {panel_num} - This is sample dialogue text for testing the Unity comic generator with interactive speech bubbles.",
            'start': i * 2.0,
            'end': (i + 1) * 2.0,
            'index': i + 1
        })
    
    # Save as SRT file
    with open('test1.srt', 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles):
            f.write(f"{i+1}\n")
            f.write(f"00:00:{i*2:02d},000 --> 00:00:{(i+1)*2:02d},000\n")
            f.write(f"{sub['text']}\n\n")
    
    print(f"✅ Created {len(subtitles)} test subtitles")
    return subtitles

def test_unity_comic_generator():
    """Test the Unity comic generator"""
    print("🚀 Testing Unity Comic Generator...")
    
    try:
        # Import the generator
        from unity_comic_generator import UnityComicGenerator
        
        # Create test data
        frames = create_test_frames()
        subtitles = create_test_subtitles()
        
        # Initialize generator
        generator = UnityComicGenerator()
        
        # Test individual methods
        print("\n📐 Testing frame resizing...")
        resized_frames = generator._resize_frames_for_unity(frames[:10])  # Test with first 10 frames
        print(f"✅ Resized {len(resized_frames)} frames")
        
        print("\n📄 Testing page generation...")
        pages_data = generator._generate_48_pages(resized_frames, subtitles)
        print(f"✅ Generated {len(pages_data)} pages")
        
        print("\n🖼️ Testing PNG page creation...")
        png_pages = generator._create_png_pages(pages_data[:3])  # Test with first 3 pages
        print(f"✅ Created {len(png_pages)} PNG pages")
        
        print("\n🌐 Testing interactive viewer...")
        generator._create_interactive_viewer(pages_data[:3])  # Test with first 3 pages
        print("✅ Created interactive viewer")
        
        print("\n💾 Testing Unity data saving...")
        generator._save_unity_data(pages_data[:3])  # Test with first 3 pages
        print("✅ Saved Unity data")
        
        print("\n🎉 Unity Comic Generator Test Complete!")
        print("📁 Check output/unity_pages/ for generated files")
        print("🌐 Open output/unity_pages/interactive_viewer.html to view")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎮 Unity Comic Generator Test")
    print("=" * 50)
    
    success = test_unity_comic_generator()
    
    if success:
        print("\n✅ All tests passed!")
        print("🎮 Unity Comic Generator is ready for use!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)