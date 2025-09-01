#!/usr/bin/env python3
"""
Run comic generation with color preservation settings
"""

import os
import sys

# Add the workspace to path
sys.path.insert(0, '/workspace')

# Import and configure the comic generator
from app_enhanced import EnhancedComicGenerator

def run_with_color_preservation():
    """Run comic generation with better color settings"""
    
    print("🎨 Running comic generation with color preservation...")
    print("-" * 50)
    
    # Create generator
    generator = EnhancedComicGenerator()
    
    # Configure for color preservation
    generator.apply_comic_style = False  # Disable comic styling to keep original colors
    generator.preserve_colors = True     # If comic styling is enabled, preserve colors
    
    print("Settings:")
    print(f"  - Comic Styling: {generator.apply_comic_style}")
    print(f"  - Color Preservation: {generator.preserve_colors}")
    print(f"  - Video Path: {generator.video_path}")
    
    # Check if video exists
    if not os.path.exists(generator.video_path):
        print(f"❌ Video not found: {generator.video_path}")
        print("Please upload a video first through the web interface")
        return
    
    # Generate comic
    print("\n🚀 Starting comic generation...")
    success = generator.generate_comic()
    
    if success:
        print("\n✅ Comic generated successfully!")
        print("📁 Check output folder for results")
        print("\nTo view:")
        print("  - Full comic: output/page.html")
        print("  - Individual panels: output/panels/")
    else:
        print("\n❌ Comic generation failed!")

if __name__ == "__main__":
    run_with_color_preservation()