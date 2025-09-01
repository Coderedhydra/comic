#!/usr/bin/env python3
"""
Comic Editor Server - Interactive bubble editing
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
from pathlib import Path

app = Flask(__name__)

# Paths
COMIC_DATA_PATH = 'output/comic_data.json'
FRAMES_DIR = 'frames/final'

@app.route('/')
def index():
    """Redirect to editor"""
    return render_template('comic_editor.html')

@app.route('/editor')
def editor():
    """Comic editor page"""
    return render_template('comic_editor.html')

@app.route('/load_comic')
def load_comic():
    """Load existing comic data"""
    try:
        # Check if we have saved data
        if os.path.exists(COMIC_DATA_PATH):
            with open(COMIC_DATA_PATH, 'r') as f:
                data = json.load(f)
                return jsonify(data)
        else:
            # Generate from existing comic
            data = generate_comic_data()
            return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_comic', methods=['POST'])
def save_comic():
    """Save comic data"""
    try:
        data = request.json
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Save JSON data
        with open(COMIC_DATA_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Generate static HTML
        generate_static_html(data)
        
        return jsonify({'success': True, 'message': 'Comic saved successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/frames/<path:filename>')
def serve_frame(filename):
    """Serve frame images"""
    return send_from_directory('frames/final', filename)

@app.route('/export_comic')
def export_comic():
    """Export comic as static HTML"""
    try:
        if os.path.exists(COMIC_DATA_PATH):
            with open(COMIC_DATA_PATH, 'r') as f:
                data = json.load(f)
                
            html = generate_static_html(data)
            return html, 200, {'Content-Type': 'text/html'}
        else:
            return "No comic data found", 404
    except Exception as e:
        return str(e), 500

def generate_comic_data():
    """Generate comic data from existing frames and subtitles"""
    # Get all frames
    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.png')])
    
    # Load subtitles if available
    subtitles = []
    if os.path.exists('test1.srt'):
        import srt
        with open('test1.srt', 'r') as f:
            subtitles = list(srt.parse(f.read()))
    
    # Create comic layout
    page_width = 800
    page_height = 1200
    panel_width = 380
    panel_height = 280
    padding = 10
    
    pages = []
    current_page = {
        'width': page_width,
        'height': page_height,
        'panels': [],
        'bubbles': []
    }
    
    # 2x2 grid layout
    positions = [
        (padding, padding),
        (page_width - panel_width - padding, padding),
        (padding, padding + panel_height + 20),
        (page_width - panel_width - padding, padding + panel_height + 20)
    ]
    
    for i, frame in enumerate(frames[:16]):  # Max 16 frames
        panel_index = i % 4
        
        # Add panel
        x, y = positions[panel_index]
        current_page['panels'].append({
            'x': x,
            'y': y,
            'width': panel_width,
            'height': panel_height,
            'image': f'/frames/{frame}'
        })
        
        # Add bubble with subtitle text
        if i < len(subtitles):
            text = subtitles[i].content.strip()
        else:
            text = f"Panel {i+1}"
            
        current_page['bubbles'].append({
            'id': f'bubble_{i}',
            'x': x + 20,
            'y': y + 20,
            'width': 150,
            'height': 60,
            'text': text,
            'panelIndex': panel_index
        })
        
        # Start new page after 4 panels
        if panel_index == 3 and i < len(frames) - 1:
            pages.append(current_page)
            current_page = {
                'width': page_width,
                'height': page_height,
                'panels': [],
                'bubbles': []
            }
    
    # Add last page
    if current_page['panels']:
        pages.append(current_page)
    
    return {'pages': pages}

def generate_static_html(data):
    """Generate static HTML from comic data"""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>My Comic</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #f0f0f0;
            font-family: Arial, sans-serif;
        }
        .comic-page {
            position: relative;
            background: white;
            margin: 20px auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .comic-panel {
            position: absolute;
            border: 2px solid #333;
            overflow: hidden;
        }
        .comic-panel img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .speech-bubble {
            position: absolute;
            background: white;
            border: 3px solid #333;
            border-radius: 20px;
            padding: 15px;
            font-family: 'Comic Sans MS', cursive;
            font-size: 14px;
            font-weight: bold;
            text-align: center;
            z-index: 10;
        }
        .bubble-tail {
            position: absolute;
            bottom: -15px;
            left: 20px;
            width: 0;
            height: 0;
            border-left: 15px solid transparent;
            border-right: 5px solid transparent;
            border-top: 20px solid #333;
            transform: rotate(-20deg);
        }
        .bubble-tail::after {
            content: '';
            position: absolute;
            bottom: 3px;
            left: -12px;
            width: 0;
            height: 0;
            border-left: 12px solid transparent;
            border-right: 4px solid transparent;
            border-top: 16px solid white;
        }
    </style>
</head>
<body>
"""
    
    for page_idx, page in enumerate(data['pages']):
        html += f'<div class="comic-page" style="width:{page["width"]}px;height:{page["height"]}px;">\n'
        
        # Add panels
        for panel in page['panels']:
            html += f'''<div class="comic-panel" style="left:{panel["x"]}px;top:{panel["y"]}px;width:{panel["width"]}px;height:{panel["height"]}px;">
                <img src="{panel["image"]}">
            </div>\n'''
        
        # Add bubbles
        for bubble in page['bubbles']:
            html += f'''<div class="speech-bubble" style="left:{bubble["x"]}px;top:{bubble["y"]}px;width:{bubble["width"]}px;height:{bubble["height"]}px;">
                {bubble["text"]}
                <div class="bubble-tail"></div>
            </div>\n'''
        
        html += '</div>\n'
    
    html += """
</body>
</html>
"""
    
    # Save to file
    with open('output/comic_static.html', 'w') as f:
        f.write(html)
    
    return html

# Integration with existing app
def add_editor_routes(existing_app):
    """Add editor routes to existing Flask app"""
    existing_app.route('/editor')(editor)
    existing_app.route('/load_comic')(load_comic)
    existing_app.route('/save_comic', methods=['POST'])(save_comic)
    existing_app.route('/export_comic')(export_comic)
    
    # Also update static JS to handle API calls
    @existing_app.route('/api/load_comic')
    def api_load_comic():
        """API endpoint for loading comic data"""
        return load_comic()
    
    print("✅ Comic editor routes added!")

if __name__ == '__main__':
    print("🎨 Starting Comic Editor Server...")
    print("📝 Visit http://localhost:5001/editor to edit your comic")
    app.run(debug=True, port=5001)