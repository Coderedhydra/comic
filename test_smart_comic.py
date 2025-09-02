#!/usr/bin/env python3
"""
Test smart comic generation with emotion matching
"""

import os
import json

def check_smart_comic():
    """Check if smart comic was generated properly"""
    
    print("🔍 Checking Smart Comic Generation")
    print("=" * 50)
    
    # Check if viewer exists
    viewer_path = 'output/smart_comic_viewer.html'
    if os.path.exists(viewer_path):
        print(f"✅ Smart comic viewer exists: {viewer_path}")
        
        # Read and check content
        with open(viewer_path, 'r') as f:
            content = f.read()
            
        # Check for image references
        import re
        images = re.findall(r'src="/frames/final/([^"]+)"', content)
        print(f"📷 Found {len(images)} image references")
        
        if images:
            print("  Images referenced:")
            for img in images[:5]:  # Show first 5
                print(f"    - {img}")
            if len(images) > 5:
                print(f"    ... and {len(images)-5} more")
        
        # Check for emotion data
        emotions = re.findall(r'emotion-(\w+)', content)
        if emotions:
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            print("\n😊 Emotion distribution:")
            for emotion, count in sorted(emotion_counts.items()):
                print(f"    {emotion}: {count}")
    else:
        print(f"❌ Smart comic viewer not found at: {viewer_path}")
    
    # Check frames directory
    print("\n📁 Checking frames directory:")
    frames_dir = 'frames/final'
    if os.path.exists(frames_dir):
        frames = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        print(f"✅ Found {len(frames)} frames in {frames_dir}")
        if frames:
            print(f"  First frame: {frames[0]}")
            print(f"  Last frame: {frames[-1]}")
    else:
        print(f"❌ Frames directory not found: {frames_dir}")
    
    # Check if we need to create a simple test
    if not os.path.exists(viewer_path) and os.path.exists(frames_dir):
        print("\n🔧 Creating a simple test smart comic...")
        create_test_smart_comic()

def create_test_smart_comic():
    """Create a test smart comic with dummy data"""
    
    # Get available frames
    frames = sorted([f for f in os.listdir('frames/final') if f.endswith('.png')])[:12]
    
    if not frames:
        print("❌ No frames available for test")
        return
    
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>Smart Comic Test</title>
    <style>
        body { margin: 20px; background: #2c3e50; color: white; font-family: Arial; }
        .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; max-width: 1000px; margin: 0 auto; }
        .panel { background: white; border: 3px solid #333; position: relative; }
        .panel img { width: 100%; height: 300px; object-fit: cover; display: block; }
        .info { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.8); color: white; padding: 10px; }
        h1 { text-align: center; }
    </style>
</head>
<body>
    <h1>🎭 Smart Comic Test</h1>
    <div class="grid">
'''
    
    test_dialogues = [
        "Hello! How are you today?",
        "I'm doing great, thanks!",
        "That's wonderful to hear!",
        "What brings you here?",
        "I'm looking for adventure!",
        "Adventure? That sounds exciting!",
        "Yes! I can't wait to start!",
        "Let me show you the way.",
        "Thank you so much!",
        "You're very welcome!",
        "This is going to be fun!",
        "Indeed it will be!"
    ]
    
    for i, frame in enumerate(frames):
        dialogue = test_dialogues[i % len(test_dialogues)]
        html += f'''
        <div class="panel">
            <img src="/frames/final/{frame}" alt="Panel {i+1}">
            <div class="info">
                <div>{dialogue}</div>
                <small>Frame: {frame}</small>
            </div>
        </div>
'''
    
    html += '''
    </div>
</body>
</html>'''
    
    os.makedirs('output', exist_ok=True)
    with open('output/smart_comic_viewer.html', 'w') as f:
        f.write(html)
    
    print("✅ Created test smart comic viewer")

if __name__ == "__main__":
    check_smart_comic()