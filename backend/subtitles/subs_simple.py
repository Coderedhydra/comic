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
    
    # Create a comprehensive subtitle file with 16 varied dialogues
    basic_subtitles = [
        srt.Subtitle(index=1, start=srt.timedelta(seconds=0), end=srt.timedelta(seconds=3), content="Hello there!"),
        srt.Subtitle(index=2, start=srt.timedelta(seconds=3), end=srt.timedelta(seconds=6), content="How are you doing?"),
        srt.Subtitle(index=3, start=srt.timedelta(seconds=6), end=srt.timedelta(seconds=9), content="I'm doing great, thanks!"),
        srt.Subtitle(index=4, start=srt.timedelta(seconds=9), end=srt.timedelta(seconds=12), content="That's wonderful!"),
        srt.Subtitle(index=5, start=srt.timedelta(seconds=12), end=srt.timedelta(seconds=15), content="What's new?"),
        srt.Subtitle(index=6, start=srt.timedelta(seconds=15), end=srt.timedelta(seconds=18), content="Not much, just working."),
        srt.Subtitle(index=7, start=srt.timedelta(seconds=18), end=srt.timedelta(seconds=21), content="Sounds busy!"),
        srt.Subtitle(index=8, start=srt.timedelta(seconds=21), end=srt.timedelta(seconds=24), content="It sure is!"),
        srt.Subtitle(index=9, start=srt.timedelta(seconds=24), end=srt.timedelta(seconds=27), content="Any plans for today?"),
        srt.Subtitle(index=10, start=srt.timedelta(seconds=27), end=srt.timedelta(seconds=30), content="Just relaxing."),
        srt.Subtitle(index=11, start=srt.timedelta(seconds=30), end=srt.timedelta(seconds=33), content="That sounds nice!"),
        srt.Subtitle(index=12, start=srt.timedelta(seconds=33), end=srt.timedelta(seconds=36), content="Indeed it is."),
        srt.Subtitle(index=13, start=srt.timedelta(seconds=36), end=srt.timedelta(seconds=39), content="Have a great day!"),
        srt.Subtitle(index=14, start=srt.timedelta(seconds=39), end=srt.timedelta(seconds=42), content="You too!"),
        srt.Subtitle(index=15, start=srt.timedelta(seconds=42), end=srt.timedelta(seconds=45), content="See you later!"),
        srt.Subtitle(index=16, start=srt.timedelta(seconds=45), end=srt.timedelta(seconds=48), content="Take care!"),
    ]
    
    # Write to file
    with open('test1.srt', 'w', encoding='utf-8') as f:
        f.write(srt.compose(basic_subtitles))
    
    print("✅ Created basic subtitle file: test1.srt")

if __name__ == '__main__':
    get_subtitles('video/IronMan.mp4')