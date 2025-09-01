#!/usr/bin/env python3
"""
Test the clean comic generation
"""

import os
import sys
import shutil

sys.path.insert(0, '/workspace')

from backend.simple_comic_generator import SimpleComicGenerator

def test_clean_generation():
    """Test clean comic generation"""
    
    print("🧪 Testing Clean Comic Generation")
    print("=" * 50)
    
    # Check if we have test data
    if not os.path.exists('test1.srt'):
        print("❌ No test subtitles found")
        print("Please generate a comic first to create test data")
        return
    
    # Create test video path
    video_path = 'video/uploaded.mp4'
    if not os.path.exists(video_path):
        print("❌ No video found at:", video_path)
        return
    
    # Test the generator
    generator = SimpleComicGenerator()
    
    print("\n📊 Configuration:")
    print(f"  - Target panels: {generator.target_panels}")
    print(f"  - Frames directory: {generator.frames_dir}")
    print(f"  - Output directory: {generator.output_dir}")
    
    print("\n🚀 Starting generation...")
    success = generator.generate_meaningful_comic(video_path)
    
    if success:
        print("\n✅ Generation successful!")
        
        # Check results
        frames = [f for f in os.listdir(generator.frames_dir) if f.endswith('.png')]
        print(f"\n📊 Results:")
        print(f"  - Frames extracted: {len(frames)}")
        print(f"  - Output location: {generator.output_dir}/comic_simple.html")
        
        # List frame files
        print("\n📸 Generated frames:")
        for i, frame in enumerate(sorted(frames)):
            print(f"  {i+1}. {frame}")
    else:
        print("\n❌ Generation failed!")

if __name__ == "__main__":
    test_clean_generation()