#!/usr/bin/env python3
"""
Test script for panel extraction feature
"""

import os
import json
from backend.panel_extractor import PanelExtractor

def test_panel_extraction():
    """Test the panel extraction functionality"""
    
    print("🧪 Testing Panel Extraction...")
    print("-" * 50)
    
    # Check if comic data exists
    if not os.path.exists("output/pages.json"):
        print("❌ No comic data found. Please generate a comic first.")
        return
    
    # Load comic data
    with open("output/pages.json", 'r') as f:
        pages_data = json.load(f)
    
    total_panels = sum(len(page.get('panels', [])) for page in pages_data)
    print(f"📊 Found {len(pages_data)} pages with {total_panels} total panels")
    
    # Create extractor
    extractor = PanelExtractor(output_dir="output/panels")
    
    # Extract panels
    print("\n📸 Extracting panels...")
    saved_panels = extractor.extract_panels_from_comic()
    
    if saved_panels:
        print(f"\n✅ Successfully extracted {len(saved_panels)} panels!")
        print(f"📁 Panels saved to: output/panels/")
        print(f"🌐 View gallery at: http://localhost:5000/panels")
        
        # Check file sizes
        print("\n📏 Panel dimensions check:")
        for i, panel_path in enumerate(saved_panels[:3]):  # Check first 3
            if os.path.exists(panel_path):
                size = os.path.getsize(panel_path) / 1024  # KB
                print(f"   - {os.path.basename(panel_path)}: {size:.1f} KB")
    else:
        print("❌ Panel extraction failed!")
    
    print("-" * 50)

if __name__ == "__main__":
    test_panel_extraction()