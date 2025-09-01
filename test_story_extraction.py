#!/usr/bin/env python3
"""
Test script for story extraction functionality
"""

import os
import json
import srt
from backend.smart_story_extractor import SmartStoryExtractor

def test_story_extraction():
    """Test the story extraction functionality"""
    
    print("🧪 Testing Story Extraction...")
    print("-" * 50)
    
    # Check if subtitles exist
    if not os.path.exists("test1.srt"):
        print("❌ No subtitles found. Please generate a comic first.")
        return
    
    # Read subtitles
    with open("test1.srt", 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))
    
    print(f"📊 Found {len(subs)} total subtitles")
    
    # Convert to JSON format
    sub_json = []
    for sub in subs:
        sub_json.append({
            'text': sub.content,
            'start': str(sub.start),
            'end': str(sub.end),
            'index': sub.index
        })
    
    # Save as JSON
    test_json = 'test_subtitles.json'
    with open(test_json, 'w') as f:
        json.dump(sub_json, f, indent=2)
    
    # Test extraction
    extractor = SmartStoryExtractor()
    
    print("\n📖 Extracting meaningful story moments...")
    meaningful = extractor.extract_meaningful_story(test_json, target_panels=12)
    
    print(f"\n✅ Selected {len(meaningful)} key moments from {len(subs)} subtitles")
    print(f"📊 Reduction: {(1 - len(meaningful)/len(subs))*100:.1f}% filtered out")
    
    # Show selected moments
    print("\n🎯 Selected Story Moments:")
    print("-" * 50)
    for i, moment in enumerate(meaningful, 1):
        print(f"{i:2d}. {moment['text'][:60]}...")
    
    # Test adaptive layout
    print("\n📐 Testing Adaptive Layout...")
    layouts = extractor.get_adaptive_layout(len(meaningful))
    
    total_panels = 0
    for i, layout in enumerate(layouts, 1):
        panels = layout['panels_per_page']
        rows = layout['rows']
        cols = layout['cols']
        total_panels += panels
        print(f"Page {i}: {rows}x{cols} grid ({panels} panels)")
    
    print(f"Total capacity: {total_panels} panels")
    
    # Show story timeline
    print("\n📚 Story Timeline:")
    timeline = extractor.create_story_timeline(meaningful)
    for phase, moments in timeline.items():
        print(f"{phase.capitalize()}: {len(moments)} panels")
    
    # Cleanup
    if os.path.exists(test_json):
        os.remove(test_json)
    
    print("-" * 50)
    print("✅ Story extraction test complete!")

if __name__ == "__main__":
    test_story_extraction()