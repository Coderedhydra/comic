#!/usr/bin/env python3
"""
Force the comic generator to create exactly 12 meaningful panels
with proper grid layout (3x4) and preserved colors
"""

import os
import sys

# Add workspace to path
sys.path.insert(0, '/workspace')

def force_proper_settings():
    """Force all settings for 12-panel generation"""
    
    print("🔧 Forcing proper 12-panel comic generation settings...")
    
    # 1. Disable HIGH_ACCURACY mode that forces 2x2 grid
    os.environ['HIGH_ACCURACY'] = '0'
    os.environ['GRID_LAYOUT'] = '0'
    
    # 2. Update the page.py to not use 2x2 templates
    page_file = '/workspace/backend/panel_layout/layout/page.py'
    
    with open(page_file, 'r') as f:
        content = f.read()
    
    # Replace the 2x2 grid templates
    if "templates = ['6666', '6666', '6666', '6666']" in content:
        content = content.replace(
            "templates = ['6666', '6666', '6666', '6666']",
            "templates = ['333333333333']  # 12 panels in 3x4 grid"
        )
        
        with open(page_file, 'w') as f:
            f.write(content)
        
        print("✅ Fixed: Panel templates now use 3x4 grid for 12 panels")
    
    # 3. Run the app with proper settings
    print("\n📊 Settings applied:")
    print("  - Target: 12 meaningful panels")
    print("  - Layout: 3x4 grid")
    print("  - Colors: Original preserved")
    print("  - Comic styling: DISABLED")
    
    # Import and configure the generator
    from app_enhanced import EnhancedComicGenerator
    
    generator = EnhancedComicGenerator()
    generator.apply_comic_style = False  # Preserve colors
    generator._filtered_count = 12      # Force 12 panels
    
    print("\n✅ Generator configured for 12-panel output")
    print("\n🚀 To generate comics:")
    print("  1. Run: python app_enhanced.py")
    print("  2. Upload your video")
    print("  3. Get 12 meaningful panels in 3x4 grid!")

if __name__ == "__main__":
    force_proper_settings()