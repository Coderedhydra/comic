#!/usr/bin/env python3
"""
Generate Smart Comic with Emotion Matching and Story Summarization
Creates a 10-15 panel comic that captures the essence of your video
"""

import os
import sys
import json

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.emotion_aware_comic import EmotionAwareComicGenerator
from backend.story_analyzer import SmartComicGenerator

def generate_smart_comic(video_path: str = None, mode: str = 'emotion'):
    """
    Generate a smart comic with emotion matching
    
    Args:
        video_path: Path to video file
        mode: 'emotion' for emotion matching, 'story' for story analysis
    """
    print("\n🎬 SMART COMIC GENERATOR")
    print("=" * 50)
    print("This will create a comic that:")
    print("• Matches facial expressions with dialogue emotions")
    print("• Summarizes long stories in 10-15 key panels")
    print("• Selects the most important story moments")
    print("• Styles speech bubbles based on emotions")
    print("=" * 50)
    
    # Check prerequisites
    if not os.path.exists('frames') and not os.path.exists('frames/final'):
        print("\n❌ No frames found! Please extract frames first.")
        print("Run: python app_enhanced.py")
        return
    
    if not os.path.exists('test1.srt'):
        print("\n❌ No subtitles found! Please generate subtitles first.")
        return
    
    # Choose generator based on mode
    if mode == 'emotion':
        print("\n🎭 Using Emotion-Aware Comic Generator...")
        generator = EmotionAwareComicGenerator()
        comic_data = generator.generate_emotion_comic(video_path or 'video/sample.mp4')
    else:
        print("\n📖 Using Story Analysis Comic Generator...")
        generator = SmartComicGenerator()
        comic_data = generator.generate_smart_comic(video_path or 'video/sample.mp4')
    
    if comic_data:
        print("\n✅ Smart comic generated successfully!")
        print("\n📊 Comic Statistics:")
        print(f"  • Total pages: {len(comic_data.get('pages', []))}")
        print(f"  • Total panels: {sum(len(p.get('panels', [])) for p in comic_data.get('pages', []))}")
        
        # Generate HTML viewer
        generate_html_viewer(comic_data)
        
        print("\n📁 Output files:")
        print("  • output/emotion_comic.json - Comic data")
        print("  • output/smart_comic_viewer.html - Interactive viewer")
        
    return comic_data

def generate_html_viewer(comic_data: Dict):
    """Generate an HTML viewer for the smart comic"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Smart Comic - Emotion Matched</title>
    <meta charset="UTF-8">
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            font-family: 'Arial', sans-serif;
            color: white;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 36px;
            margin: 0;
            color: #fff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .header p {
            color: #ccc;
            margin-top: 10px;
        }
        
        .comic-container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .comic-page {
            position: relative;
            background: white;
            margin: 30px auto;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            border-radius: 10px;
            overflow: hidden;
        }
        
        .comic-panel {
            position: absolute;
            border: 3px solid #333;
            overflow: hidden;
            transition: transform 0.2s;
        }
        
        .comic-panel:hover {
            transform: scale(1.02);
            z-index: 10;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }
        
        .comic-panel img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .speech-bubble {
            position: absolute;
            border-radius: 20px;
            padding: 12px 18px;
            font-family: 'Comic Sans MS', cursive;
            font-weight: bold;
            text-align: center;
            z-index: 20;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
            transition: all 0.3s;
        }
        
        .speech-bubble:hover {
            transform: scale(1.05);
            box-shadow: 4px 4px 12px rgba(0,0,0,0.3);
        }
        
        /* Emotion-specific styles */
        .emotion-happy {
            border: 3px solid #4CAF50;
            background: #E8F5E9;
            animation: bounce 2s infinite;
        }
        
        .emotion-sad {
            border: 3px solid #2196F3;
            background: #E3F2FD;
            transform: translateY(5px);
        }
        
        .emotion-angry {
            border: 4px solid #F44336;
            background: #FFEBEE;
            font-size: 16px;
            animation: shake 0.5s infinite;
        }
        
        .emotion-surprised {
            border: 3px solid #FF9800;
            background: #FFF3E0;
            font-size: 16px;
            transform: scale(1.1);
        }
        
        .emotion-neutral {
            border: 2px solid #333;
            background: #FFF;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-2px); }
            75% { transform: translateX(2px); }
        }
        
        .emotion-indicator {
            position: absolute;
            top: 5px;
            right: 5px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 10px;
            text-transform: uppercase;
        }
        
        .page-number {
            text-align: center;
            margin-top: 10px;
            color: #666;
            font-size: 14px;
        }
        
        .stats {
            background: #2c3e50;
            padding: 20px;
            border-radius: 10px;
            margin: 30px auto;
            max-width: 900px;
            text-align: center;
        }
        
        .stats h3 {
            margin-top: 0;
            color: #3498db;
        }
        
        .emotion-legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
        }
        
        .emotion-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .emotion-dot {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎭 Smart Comic with Emotion Matching</h1>
        <p>AI-generated comic that matches facial expressions with dialogue emotions</p>
    </div>
    
    <div class="stats">
        <h3>📊 Emotion Distribution</h3>
        <div class="emotion-legend">
            <div class="emotion-item">
                <div class="emotion-dot" style="background: #E8F5E9; border-color: #4CAF50;"></div>
                <span>Happy</span>
            </div>
            <div class="emotion-item">
                <div class="emotion-dot" style="background: #E3F2FD; border-color: #2196F3;"></div>
                <span>Sad</span>
            </div>
            <div class="emotion-item">
                <div class="emotion-dot" style="background: #FFEBEE; border-color: #F44336;"></div>
                <span>Angry</span>
            </div>
            <div class="emotion-item">
                <div class="emotion-dot" style="background: #FFF3E0; border-color: #FF9800;"></div>
                <span>Surprised</span>
            </div>
        </div>
    </div>
    
    <div class="comic-container">
"""
    
    # Add pages
    for page_num, page in enumerate(comic_data.get('pages', [])):
        html += f'<div class="comic-page" style="width:{page["width"]}px;height:{page["height"]}px;">\n'
        
        # Add panels
        for panel in page.get('panels', []):
            emotion = panel.get('emotion', 'neutral')
            html += f'<div class="comic-panel" style="left:{panel["x"]}px;top:{panel["y"]}px;width:{panel["width"]}px;height:{panel["height"]}px;">\n'
            html += f'<img src="{panel["image"]}" alt="Panel">\n'
            html += f'<div class="emotion-indicator">{emotion}</div>\n'
            html += '</div>\n'
        
        # Add bubbles
        for bubble in page.get('bubbles', []):
            style = bubble.get('style', {})
            emotion_class = f"emotion-{style.get('emotion', 'neutral')}"
            
            html += f'<div class="speech-bubble {emotion_class}" style="'
            html += f'left:{bubble["x"]}px;top:{bubble["y"]}px;'
            html += f'width:{bubble.get("width", 150)}px;min-height:{bubble.get("height", 60)}px;'
            
            if 'border' in style:
                html += f'border-color:{style["border"]};'
            if 'background' in style:
                html += f'background:{style["background"]};'
                
            html += '">'
            html += bubble["text"]
            html += '</div>\n'
        
        html += '</div>\n'
        html += f'<div class="page-number">Page {page_num + 1}</div>\n'
    
    html += """
    </div>
</body>
</html>
"""
    
    # Save HTML
    os.makedirs('output', exist_ok=True)
    with open('output/smart_comic_viewer.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print("✅ Generated HTML viewer: output/smart_comic_viewer.html")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate smart comic with emotion matching')
    parser.add_argument('video', nargs='?', help='Path to video file')
    parser.add_argument('--mode', choices=['emotion', 'story'], default='emotion',
                       help='Generation mode: emotion matching or story analysis')
    parser.add_argument('--panels', type=int, default=12,
                       help='Target number of panels (10-15)')
    
    args = parser.parse_args()
    
    # Generate smart comic
    generate_smart_comic(args.video, args.mode)

if __name__ == "__main__":
    main()