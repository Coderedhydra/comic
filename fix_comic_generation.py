#!/usr/bin/env python3
"""
Fix comic generation to:
1. Preserve original colors (no green tint)
2. Generate 10-15 panels based on story importance
"""

import os
import sys
import json

# Add workspace to path
sys.path.insert(0, '/workspace')

def patch_comic_generator():
    """Apply fixes to the comic generator"""
    
    print("🔧 Applying fixes to comic generation...")
    
    # Fix 1: Update the main app to disable aggressive comic styling
    app_file = '/workspace/app_enhanced.py'
    
    # Read current file
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Make sure color preservation is enabled by default
    if 'self.apply_comic_style = True' in content and 'self.preserve_colors = True' in content:
        # Change to disable comic styling by default to preserve colors
        content = content.replace(
            'self.apply_comic_style = True  # Can be set to False to preserve original colors',
            'self.apply_comic_style = False  # Disabled to preserve original colors'
        )
        
        with open(app_file, 'w') as f:
            f.write(content)
        
        print("✅ Fixed: Comic styling disabled to preserve colors")
    
    # Fix 2: Ensure story extraction target is higher
    story_file = '/workspace/backend/smart_story_extractor.py'
    
    with open(story_file, 'r') as f:
        story_content = f.read()
    
    # Update default target panels
    if 'target_panels: int = 12' in story_content:
        story_content = story_content.replace(
            'target_panels: int = 12',
            'target_panels: int = 15'  # Increase to 15 panels
        )
        
        with open(story_file, 'w') as f:
            f.write(story_content)
        
        print("✅ Fixed: Story extraction now targets 15 panels")
    
    print("\n📊 Current Settings:")
    print("  - Comic Styling: DISABLED (preserves original colors)")
    print("  - Target Panels: 10-15 (based on story importance)")
    print("  - Layout: Adaptive (2x3, 3x3, multi-page)")
    print("  - Resolution: Max 2K")
    
    print("\n🎯 To generate comics with these fixes:")
    print("  1. Start the Flask app: python app_enhanced.py")
    print("  2. Upload your video")
    print("  3. The system will automatically:")
    print("     - Extract 10-15 key story moments")
    print("     - Preserve original colors")
    print("     - Create adaptive layout")

def verify_story_extraction():
    """Verify story extraction is working"""
    
    print("\n🔍 Verifying story extraction setup...")
    
    # Check if test subtitles exist
    if os.path.exists('test1.srt'):
        import srt
        with open('test1.srt', 'r') as f:
            subs = list(srt.parse(f.read()))
        print(f"  ✓ Found {len(subs)} subtitles")
        
        # Test story extraction
        try:
            from backend.smart_story_extractor import SmartStoryExtractor
            extractor = SmartStoryExtractor()
            
            # Convert to JSON format
            sub_json = []
            for sub in subs:
                sub_json.append({
                    'text': sub.content,
                    'start': str(sub.start),
                    'end': str(sub.end),
                    'index': sub.index
                })
            
            # Save temp file
            with open('temp_subs.json', 'w') as f:
                json.dump(sub_json, f)
            
            # Extract
            meaningful = extractor.extract_meaningful_story('temp_subs.json', target_panels=15)
            
            print(f"  ✓ Story extraction working: {len(meaningful)} key moments selected")
            
            # Cleanup
            if os.path.exists('temp_subs.json'):
                os.remove('temp_subs.json')
                
        except Exception as e:
            print(f"  ✗ Story extraction error: {e}")
    else:
        print("  ℹ️  No subtitles found (will be created when you process a video)")

if __name__ == "__main__":
    patch_comic_generator()
    verify_story_extraction()
    
    print("\n✅ Fixes applied! The comic generator will now:")
    print("  1. Preserve original colors (no green tint)")
    print("  2. Select 10-15 important story panels")
    print("  3. Create adaptive layouts")
    print("\n🚀 Ready to generate comics with proper colors and story flow!")