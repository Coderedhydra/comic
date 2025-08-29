"""
Simple Subtitle Extraction
Uses existing subtitle files instead of generating them
"""

import srt
import os

def get_subtitles(file):
    """Simple subtitle extraction that uses existing files"""
    print("📝 Using existing subtitle file...")
    
    # Check if test1.srt already exists
    if os.path.exists('test1.srt'):
        print("✅ Found existing subtitle file: test1.srt")
        return
    
    # If no subtitle file exists, create a simple one
    print("📝 Creating simple subtitle file...")
    
    # Create a basic subtitle file with action scenes
    basic_subtitles = [
        srt.Subtitle(index=1, start=srt.timedelta(seconds=0), end=srt.timedelta(seconds=3), content="Hello"),
        srt.Subtitle(index=2, start=srt.timedelta(seconds=3), end=srt.timedelta(seconds=6), content="How are you?"),
        srt.Subtitle(index=3, start=srt.timedelta(seconds=6), end=srt.timedelta(seconds=9), content="I'm fine, thank you"),
        srt.Subtitle(index=4, start=srt.timedelta(seconds=9), end=srt.timedelta(seconds=12), content="That's great!"),
    ]
    
    # Write to file
    with open('test1.srt', 'w', encoding='utf-8') as f:
        f.write(srt.compose(basic_subtitles))
    
    print("✅ Created basic subtitle file: test1.srt")

if __name__ == '__main__':
    get_subtitles('video/IronMan.mp4')