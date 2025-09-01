#!/usr/bin/env python3
"""
Run the Comic Editor
Allows dragging bubbles and editing text
"""

import os
import sys
import json
import webbrowser
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('comic_editor.html')

@app.route('/editor')
def editor():
    return render_template('comic_editor.html')

@app.route('/load_comic')
def load_comic():
    """Load comic data"""
    try:
        # First check if we have a saved comic
        if os.path.exists('output/comic_data.json'):
            with open('output/comic_data.json', 'r') as f:
                return jsonify(json.load(f))
        
        # Otherwise, generate from existing frames
        return jsonify(generate_from_frames())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_comic', methods=['POST'])
def save_comic():
    """Save edited comic"""
    try:
        data = request.json
        os.makedirs('output', exist_ok=True)
        
        # Save JSON
        with open('output/comic_data.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        # Generate HTML
        generate_html_output(data)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/frames/<path:filename>')
def serve_frame(filename):
    """Serve frame images"""
    frames_dir = 'frames/final'
    if not os.path.exists(os.path.join(frames_dir, filename)):
        frames_dir = 'frames'
    return send_from_directory(frames_dir, filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

def generate_from_frames():
    """Generate comic layout from existing frames"""
    frames_dir = 'frames/final' if os.path.exists('frames/final') else 'frames'
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])[:16]
    
    # Load subtitles
    subtitles = []
    if os.path.exists('test1.srt'):
        try:
            import srt
            with open('test1.srt', 'r') as f:
                subtitles = list(srt.parse(f.read()))
        except:
            pass
    
    # Create 2x2 grid layout
    pages = []
    page_width = 800
    page_height = 600
    
    for page_num in range(0, len(frames), 4):
        page = {
            'width': page_width,
            'height': page_height,
            'panels': [],
            'bubbles': []
        }
        
        # Panel positions for 2x2 grid
        positions = [
            (10, 10, 380, 280),    # Top left
            (410, 10, 380, 280),   # Top right
            (10, 310, 380, 280),   # Bottom left
            (410, 310, 380, 280)   # Bottom right
        ]
        
        for i in range(4):
            frame_idx = page_num + i
            if frame_idx >= len(frames):
                break
                
            x, y, w, h = positions[i]
            
            # Add panel
            page['panels'].append({
                'x': x, 'y': y,
                'width': w, 'height': h,
                'image': f'/frames/{frames[frame_idx]}'
            })
            
            # Add bubble with subtitle or default text
            text = "Click to edit"
            if frame_idx < len(subtitles):
                text = subtitles[frame_idx].content.strip()[:50]  # Limit length
            
            page['bubbles'].append({
                'id': f'bubble_{frame_idx}',
                'x': x + 20,
                'y': y + 20,
                'width': 150,
                'height': 60,
                'text': text
            })
        
        pages.append(page)
    
    return {'pages': pages}

def generate_html_output(data):
    """Generate static HTML file"""
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>My Comic</title>
    <style>
        body { margin: 0; padding: 20px; background: #f0f0f0; }
        .comic-page { position: relative; background: white; margin: 20px auto; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .comic-panel { position: absolute; border: 2px solid #333; overflow: hidden; }
        .comic-panel img { width: 100%; height: 100%; object-fit: cover; }
        .speech-bubble { position: absolute; background: white; border: 3px solid #333; border-radius: 20px; padding: 15px; font-family: "Comic Sans MS", cursive; font-size: 14px; font-weight: bold; text-align: center; z-index: 10; }
        .print-page { page-break-after: always; }
        @media print { body { margin: 0; padding: 0; } .comic-page { box-shadow: none; } }
    </style>
</head>
<body>
'''
    
    for page in data['pages']:
        html += f'<div class="comic-page print-page" style="width:{page["width"]}px;height:{page["height"]}px;">\n'
        
        for panel in page['panels']:
            html += f'<div class="comic-panel" style="left:{panel["x"]}px;top:{panel["y"]}px;width:{panel["width"]}px;height:{panel["height"]}px;"><img src="{panel["image"]}"></div>\n'
        
        for bubble in page['bubbles']:
            html += f'<div class="speech-bubble" style="left:{bubble["x"]}px;top:{bubble["y"]}px;width:{bubble["width"]}px;height:{bubble["height"]}px;">{bubble["text"]}</div>\n'
        
        html += '</div>\n'
    
    html += '</body></html>'
    
    with open('output/comic_final.html', 'w') as f:
        f.write(html)
    
    print("✅ Saved comic to output/comic_final.html")

if __name__ == '__main__':
    print("\n🎨 COMIC EDITOR")
    print("=" * 50)
    print("This editor allows you to:")
    print("• Drag speech bubbles to reposition them")
    print("• Double-click bubbles to edit text")
    print("• Add new bubbles with the toolbar")
    print("• Save and export your edited comic")
    print("=" * 50)
    
    # Check if frames exist
    if not os.path.exists('frames') and not os.path.exists('frames/final'):
        print("\n❌ No frames found! Please generate a comic first.")
        sys.exit(1)
    
    port = 5001
    url = f'http://localhost:{port}/editor'
    
    print(f"\n🚀 Starting editor server on port {port}...")
    print(f"📝 Opening browser to: {url}")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Open browser after a short delay
    import threading
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    
    app.run(debug=False, port=port)