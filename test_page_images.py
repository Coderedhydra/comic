#!/usr/bin/env python3
"""Test page image generation"""

import os
import json
from backend.page_image_generator import generate_page_images_from_json

def test_page_image_generation():
    """Test generating page images from existing comic"""
    
    # Check if we have a pages.json file
    pages_json = "output/pages.json"
    if not os.path.exists(pages_json):
        print("❌ No pages.json found. Please generate a comic first.")
        return
    
    # Generate page images
    print("📄 Generating page images at 800x1080...")
    images = generate_page_images_from_json(
        json_path=pages_json,
        frames_dir="frames/final",
        output_dir="output/page_images"
    )
    
    if images:
        print(f"\n✅ Successfully generated {len(images)} page images!")
        print("\n📁 Files saved to: output/page_images/")
        print("🌐 View gallery at: output/page_images/index.html")
        
        # List the generated files
        print("\n📄 Generated files:")
        for img in images:
            size_kb = os.path.getsize(img) / 1024
            print(f"  - {os.path.basename(img)} ({size_kb:.1f} KB)")
    else:
        print("❌ No images were generated")

if __name__ == "__main__":
    test_page_image_generation()