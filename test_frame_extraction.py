#!/usr/bin/env python3
"""
Test frame extraction to understand image dimensions
"""

import os
import sys

# Add the current directory to Python path
sys.path.append('.')

def test_frame_extraction():
    """Test extracting a single frame to see dimensions"""
    
    video_path = "video/IronMan.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"🎬 Testing frame extraction from: {video_path}")
    
    try:
        # Try to import cv2
        import cv2
        print("✅ OpenCV available")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ Could not open video")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"📐 Video dimensions: {width}x{height}")
        print(f"🎞️ FPS: {fps}")
        print(f"⏱️ Duration: {duration:.2f} seconds")
        print(f"🎬 Total frames: {frame_count}")
        
        # Extract first frame
        ret, frame = cap.read()
        if ret:
            # Create test directory
            test_dir = "test_frames"
            os.makedirs(test_dir, exist_ok=True)
            
            # Save frame
            frame_path = os.path.join(test_dir, "test_frame.png")
            cv2.imwrite(frame_path, frame)
            
            # Check saved frame dimensions
            saved_img = cv2.imread(frame_path)
            if saved_img is not None:
                saved_height, saved_width = saved_img.shape[:2]
                print(f"💾 Saved frame dimensions: {saved_width}x{saved_height}")
                print(f"📁 Frame saved to: {frame_path}")
                
                # Check file size
                file_size = os.path.getsize(frame_path)
                print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            else:
                print("❌ Could not read saved frame")
        else:
            print("❌ Could not read frame from video")
        
        cap.release()
        
    except ImportError:
        print("❌ OpenCV not available")
        print("💡 Install with: pip install opencv-python")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_simple_extraction():
    """Test the actual extraction process used in the app"""
    
    print("\n🔧 Testing actual extraction process...")
    
    try:
        from backend.keyframes.extract_frames import extract_frames
        
        video_path = "video/IronMan.mp4"
        output_path = "test_frames/extract_test"
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Extract frames from first 5 seconds
        print("📸 Extracting frames from first 5 seconds...")
        frames = extract_frames(video_path, output_path, 0, 5, 1)  # 1 frame per second
        
        print(f"✅ Extracted {len(frames)} frames")
        
        # Check dimensions of extracted frames
        for i, frame_path in enumerate(frames):
            if os.path.exists(frame_path):
                try:
                    import cv2
                    img = cv2.imread(frame_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        file_size = os.path.getsize(frame_path)
                        print(f"  Frame {i+1}: {width}x{height} ({file_size/1024:.1f} KB)")
                except:
                    print(f"  Frame {i+1}: Could not read dimensions")
        
    except Exception as e:
        print(f"❌ Extraction test failed: {e}")

if __name__ == "__main__":
    test_frame_extraction()
    test_simple_extraction()