"""
Package comic as self-contained HTML file with all editing features
"""

import os
import base64
import json
from pathlib import Path

def create_portable_comic(pages_json_path="output/pages.json", output_path="output/comic_portable.html"):
    """
    Create a single HTML file that contains everything:
    - All images embedded as base64
    - All editing functionality
    - No external dependencies
    """
    
    # Read pages data
    with open(pages_json_path, 'r') as f:
        pages_data = json.load(f)
    
    # Convert all images to base64
    embedded_images = {}
    frames_dir = "frames/final"
    
    for page in pages_data:
        for panel in page.get('panels', []):
            img_name = panel.get('image', '')
            if img_name and img_name not in embedded_images:
                img_path = os.path.join(frames_dir, img_name)
                if os.path.exists(img_path):
                    with open(img_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        embedded_images[img_name] = f"data:image/png;base64,{img_data}"
    
    # Create self-contained HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portable Comic Editor</title>
    <style>
        body {{ margin: 0; padding: 20px; background: #f0f0f0; font-family: Arial, sans-serif; }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        .header h1 {{ color: #333; }}
        .save-notice {{ background: #4CAF50; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .comic-container {{ max-width: 1200px; margin: 0 auto; }}
        .comic-page {{ background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .comic-grid {{ display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 10px; height: 600px; }}
        .panel {{ position: relative; border: 2px solid #333; overflow: hidden; }}
        .panel img {{ width: 100%; height: 100%; object-fit: cover; }}
        .speech-bubble {{ 
            position: absolute; 
            background: white; 
            border: 3px solid #333; 
            border-radius: 15px; 
            padding: 12px; 
            max-width: 200px; 
            font-size: 14px; 
            font-weight: bold;
            box-shadow: 3px 3px 8px rgba(0,0,0,0.4);
            z-index: 10;
            text-align: center;
            color: #333;
            cursor: move;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .speech-bubble:hover {{ 
            transform: scale(1.02); 
            box-shadow: 3px 3px 12px rgba(0,0,0,0.6); 
        }}
        .speech-bubble.editing {{ cursor: text; }}
        .speech-bubble textarea {{
            width: 100%;
            height: 100%;
            border: none;
            background: transparent;
            font: inherit;
            text-align: center;
            resize: none;
            outline: 2px solid #4CAF50;
            padding: 5px;
        }}
        .edit-controls {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.85);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            font-size: 14px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .edit-controls h4 {{ margin: 0 0 10px 0; color: #4CAF50; }}
        .edit-controls p {{ margin: 5px 0; opacity: 0.9; }}
        .edit-controls button {{
            margin-top: 10px;
            padding: 8px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            width: 100%;
        }}
        .save-html-btn {{ background: #4CAF50; color: white; }}
        .save-html-btn:hover {{ background: #45a049; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Portable Comic Editor</h1>
        <div class="save-notice">
            💡 This is a self-contained file! Save this HTML to keep your comic and edits.
            <br>You can open and edit it anytime in any browser.
        </div>
    </div>
    
    <div class="comic-container">
        <div id="comic-pages"></div>
    </div>
    
    <div class="edit-controls">
        <h4>✏️ Interactive Editor</h4>
        <p>• <strong>Drag</strong> bubbles to move</p>
        <p>• <strong>Double-click</strong> to edit text</p>
        <p>• <strong>Save</strong> this HTML file to keep edits!</p>
        <button class="save-html-btn" onclick="saveThisFile()">💾 Download Updated HTML</button>
    </div>

    <script>
        // Embedded pages data
        const pagesData = {json.dumps(pages_data)};
        
        // Embedded images
        const embeddedImages = {json.dumps(embedded_images)};
        
        // Render comic
        function renderComic() {{
            const container = document.getElementById('comic-pages');
            container.innerHTML = '';
            
            pagesData.forEach((page, pageIndex) => {{
                const pageDiv = document.createElement('div');
                pageDiv.className = 'comic-page';
                
                const gridDiv = document.createElement('div');
                gridDiv.className = 'comic-grid';
                
                page.panels.forEach((panel, panelIndex) => {{
                    const panelDiv = document.createElement('div');
                    panelDiv.className = 'panel';
                    
                    const img = document.createElement('img');
                    img.src = embeddedImages[panel.image] || panel.image;
                    panelDiv.appendChild(img);
                    
                    gridDiv.appendChild(panelDiv);
                }});
                
                pageDiv.appendChild(gridDiv);
                
                // Add bubbles
                page.bubbles.forEach((bubble, bubbleIndex) => {{
                    const bubbleDiv = document.createElement('div');
                    bubbleDiv.className = 'speech-bubble';
                    bubbleDiv.style.left = bubble.bubble_offset_x + 'px';
                    bubbleDiv.style.top = bubble.bubble_offset_y + 'px';
                    bubbleDiv.innerText = bubble.dialog;
                    
                    const targetPanel = gridDiv.children[Math.floor(bubbleIndex / 2)];
                    if (targetPanel) {{
                        targetPanel.appendChild(bubbleDiv);
                    }}
                }});
                
                container.appendChild(pageDiv);
            }});
            
            // Initialize editor
            setTimeout(initializeEditor, 100);
        }}
        
        // Editor functionality (same as before)
        let currentEditBubble = null;
        let draggedBubble = null;
        let offset = {{x: 0, y: 0}};
        
        function initializeEditor() {{
            document.querySelectorAll('.speech-bubble').forEach(bubble => {{
                bubble.addEventListener('dblclick', (e) => {{
                    e.stopPropagation();
                    editBubbleText(bubble);
                }});
                bubble.addEventListener('mousedown', startDrag);
            }});
            
            document.addEventListener('mousemove', drag);
            document.addEventListener('mouseup', stopDrag);
        }}
        
        function editBubbleText(bubble) {{
            if (currentEditBubble) return;
            
            currentEditBubble = bubble;
            bubble.classList.add('editing');
            
            const text = bubble.innerText;
            const textarea = document.createElement('textarea');
            textarea.value = text;
            
            bubble.innerHTML = '';
            bubble.appendChild(textarea);
            textarea.focus();
            textarea.select();
            
            textarea.addEventListener('keydown', (e) => {{
                if (e.key === 'Enter' && !e.shiftKey) {{
                    e.preventDefault();
                    saveBubbleText(bubble, textarea.value);
                }}
                if (e.key === 'Escape') {{
                    saveBubbleText(bubble, text);
                }}
            }});
            
            textarea.addEventListener('blur', () => {{
                setTimeout(() => {{
                    if (currentEditBubble === bubble) {{
                        saveBubbleText(bubble, textarea.value);
                    }}
                }}, 100);
            }});
        }}
        
        function saveBubbleText(bubble, text) {{
            bubble.innerText = text;
            bubble.classList.remove('editing');
            currentEditBubble = null;
        }}
        
        function startDrag(e) {{
            if (e.target.tagName === 'TEXTAREA') return;
            
            const bubble = e.target.closest('.speech-bubble');
            if (!bubble || currentEditBubble) return;
            
            draggedBubble = bubble;
            const rect = bubble.getBoundingClientRect();
            offset.x = e.clientX - rect.left;
            offset.y = e.clientY - rect.top;
            
            bubble.style.opacity = '0.9';
            bubble.style.zIndex = '100';
            e.preventDefault();
        }}
        
        function drag(e) {{
            if (!draggedBubble) return;
            
            const parent = draggedBubble.parentElement;
            const parentRect = parent.getBoundingClientRect();
            
            let x = e.clientX - parentRect.left - offset.x;
            let y = e.clientY - parentRect.top - offset.y;
            
            x = Math.max(0, Math.min(x, parentRect.width - draggedBubble.offsetWidth));
            y = Math.max(0, Math.min(y, parentRect.height - draggedBubble.offsetHeight));
            
            draggedBubble.style.left = x + 'px';
            draggedBubble.style.top = y + 'px';
        }}
        
        function stopDrag() {{
            if (draggedBubble) {{
                draggedBubble.style.opacity = '';
                draggedBubble.style.zIndex = '';
                draggedBubble = null;
            }}
        }}
        
        function saveThisFile() {{
            // Get current state
            const currentHTML = document.documentElement.outerHTML;
            
            // Create blob and download
            const blob = new Blob([currentHTML], {{type: 'text/html'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'comic_editable_' + new Date().getTime() + '.html';
            a.click();
            URL.revokeObjectURL(url);
        }}
        
        // Initialize on load
        renderComic();
    </script>
</body>
</html>'''
    
    # Write the file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path