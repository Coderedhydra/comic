#!/usr/bin/env python3
"""
Test 12-page comic generation
"""

import os
import sys

sys.path.insert(0, '/workspace')

# Mock the class_def to avoid import issues
class MockPanel:
    def __init__(self, image, row_span, col_span):
        self.image = image
        self.row_span = row_span
        self.col_span = col_span

class MockPage:
    def __init__(self, panels, bubbles):
        self.panels = panels
        self.bubbles = bubbles

# Replace the imports
import backend.fixed_12_pages_2x2 as fixed_pages
backend.fixed_12_pages_2x2.panel = MockPanel
backend.fixed_12_pages_2x2.Page = MockPage

def test_12_page_generation():
    """Test the 12-page generation logic"""
    
    print("🧪 Testing 12-page comic generation")
    print("=" * 50)
    
    # Create mock frames
    test_frames = [f"frame{i:03d}.png" for i in range(50)]  # 50 test frames
    test_bubbles = []
    
    print(f"📊 Test data: {len(test_frames)} frames")
    
    # Test the generation
    pages = fixed_pages.generate_12_pages_2x2_grid(test_frames, test_bubbles)
    
    print(f"\n✅ Generated {len(pages)} pages")
    
    # Verify structure
    total_panels = 0
    for i, page in enumerate(pages):
        panels_on_page = len(page.panels)
        total_panels += panels_on_page
        print(f"  Page {i+1}: {panels_on_page} panels")
    
    print(f"\n📊 Total panels: {total_panels}")
    print(f"📐 Grid: 2x2 per page")
    
    # Test frame selection
    selected = fixed_pages.select_meaningful_frames(test_frames, 48)
    print(f"\n🎯 Frame selection: {len(selected)} meaningful frames from {len(test_frames)}")

if __name__ == "__main__":
    test_12_page_generation()