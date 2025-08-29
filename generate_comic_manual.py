#!/usr/bin/env python3
"""
Manual Comic Generator
Generate comic directly from existing video file
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

def generate_comic_from_video():
    """Generate comic from existing video file"""
    print("🎬 Manual Comic Generator")
    print("=" * 40)
    
    # Check if video exists
    video_path = 'video/IronMan.mp4'
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return False
    
    print(f"✅ Found video: {video_path}")
    
    try:
        # Import the comic generator
        from app_enhanced import comic_generator
        
        # Set video path
        comic_generator.video_path = video_path
        
        # Generate comic
        print("🚀 Starting comic generation...")
        success = comic_generator.generate_comic()
        
        if success:
            # Check if output was created
            output_path = 'output/page.html'
            if os.path.exists(output_path):
                print(f"✅ Comic generated successfully!")
                print(f"📁 Location: {os.path.abspath(output_path)}")
                
                # Try to open in browser
                try:
                    print("🌐 Opening in browser...")
                    webbrowser.open(f'file://{os.path.abspath(output_path)}')
                    print("✅ Browser opened!")
                except Exception as e:
                    print(f"⚠️ Could not open browser: {e}")
                    print(f"📁 Please open manually: {output_path}")
                
                return True
            else:
                print("❌ Output file not found")
                return False
        else:
            print("❌ Comic generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = generate_comic_from_video()
    
    if success:
        print("\n🎉 Comic generation completed!")
        print("📁 Check the output folder for your comic")
    else:
        print("\n❌ Comic generation failed")
        print("🔍 Run test_comic_generation.py to debug")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)