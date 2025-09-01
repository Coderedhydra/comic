#!/usr/bin/env python3
"""
Quick test for advanced image enhancer
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.advanced_image_enhancer import get_advanced_enhancer

def test_enhancer():
    """Test the advanced enhancer"""
    print("🧪 Testing Advanced Image Enhancer...")
    
    # Get enhancer
    enhancer = get_advanced_enhancer()
    
    # Check if we have any frames to test
    frames_dir = "frames/final"
    if os.path.exists(frames_dir):
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        if frame_files:
            # Test with first frame
            test_frame = os.path.join(frames_dir, frame_files[0])
            print(f"📸 Testing with: {test_frame}")
            
            # Create test output
            test_output = os.path.join(frames_dir, f"test_enhanced_{frame_files[0]}")
            
            # Apply enhancement
            result = enhancer.enhance_image(test_frame, test_output)
            
            if result != test_frame:
                print(f"✅ Enhancement successful: {result}")
                
                # Check file size
                original_size = os.path.getsize(test_frame)
                enhanced_size = os.path.getsize(result)
                print(f"📊 Original size: {original_size:,} bytes")
                print(f"📊 Enhanced size: {enhanced_size:,} bytes")
                print(f"📈 Size increase: {((enhanced_size/original_size)-1)*100:.1f}%")
            else:
                print("❌ Enhancement failed")
        else:
            print("❌ No frames found for testing")
    else:
        print("❌ Frames directory not found")
    
    print("✅ Advanced enhancer test completed!")

if __name__ == "__main__":
    test_enhancer()