#!/usr/bin/env python3
"""
Run Unity Comic Generator
Creates 48 pages with 2x2 grid layout for Unity integration
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate Unity Comic with 48 pages')
    parser.add_argument('--video', '-v', help='Path to video file', default='video/uploaded.mp4')
    parser.add_argument('--test', '-t', action='store_true', help='Run test mode with sample data')
    parser.add_argument('--pages', '-p', type=int, default=48, help='Number of pages to generate (default: 48)')
    
    args = parser.parse_args()
    
    print("🎮 Unity Comic Generator")
    print("=" * 50)
    
    if args.test:
        print("🧪 Running in test mode...")
        try:
            from minimal_unity_test import create_minimal_test
            success = create_minimal_test()
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return 1
    else:
        print(f"🎬 Generating {args.pages} pages from video: {args.video}")
        try:
            from unity_comic_generator import UnityComicGenerator
            
            generator = UnityComicGenerator()
            generator.total_pages = args.pages
            
            success = generator.generate_48_pages_comic(args.video)
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    if success:
        print("\n🎉 Unity Comic Generation Complete!")
        print("📁 Check output/unity_pages/ for all files")
        print("🌐 Open output/unity_pages/interactive_viewer.html to view")
        print("📖 Read output/unity_pages/UNITY_INTEGRATION.md for Unity setup")
        return 0
    else:
        print("\n❌ Unity Comic Generation Failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())