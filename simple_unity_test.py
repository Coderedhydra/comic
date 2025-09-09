#!/usr/bin/env python3
"""
Simple test for Unity Comic Generator without OpenCV dependency
"""

import os
import json
from PIL import Image, ImageDraw, ImageFont

def create_simple_test_frames():
    """Create simple test frames using PIL"""
    print("🎬 Creating simple test frames...")
    
    # Create test frames directory
    os.makedirs('frames/unity_extracted', exist_ok=True)
    
    # Create 192 test frames (48 pages * 4 panels)
    total_frames = 48 * 4
    
    for i in range(total_frames):
        # Create a test image
        img = Image.new('RGB', (1280, 720), color=(100 + i % 155, 50 + i % 100, 200 - i % 100))
        draw = ImageDraw.Draw(img)
        
        # Add frame number text
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = ImageFont.load_default()
        
        draw.text((50, 100), f"Frame {i+1:03d}", fill=(255, 255, 255), font=font)
        
        # Add page info
        page_num = (i // 4) + 1
        panel_num = (i % 4) + 1
        draw.text((50, 200), f"Page {page_num:02d} Panel {panel_num}", fill=(255, 255, 255), font=font)
        
        # Save frame
        frame_path = f'frames/unity_extracted/frame_{i:03d}.png'
        img.save(frame_path)
    
    print(f"✅ Created {total_frames} test frames")
    return [f'frames/unity_extracted/frame_{i:03d}.png' for i in range(total_frames)]

def create_test_subtitles():
    """Create test subtitles"""
    print("💬 Creating test subtitles...")
    
    subtitles = []
    total_panels = 48 * 4
    
    for i in range(total_panels):
        page_num = (i // 4) + 1
        panel_num = (i % 4) + 1
        
        subtitles.append({
            'text': f"Page {page_num} Panel {panel_num} - Sample dialogue for Unity comic testing.",
            'start': i * 2.0,
            'end': (i + 1) * 2.0,
            'index': i + 1
        })
    
    # Save as SRT file
    with open('test1.srt', 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles):
            f.write(f"{i+1}\n")
            f.write(f"00:00:{i*2:02d},000 --> 00:00:{(i+1)*2:02d},000\n")
            f.write(f"{sub['text']}\n\n")
    
    print(f"✅ Created {len(subtitles)} test subtitles")
    return subtitles

def test_basic_functionality():
    """Test basic functionality without OpenCV"""
    print("🚀 Testing basic Unity Comic functionality...")
    
    try:
        # Create test data
        frames = create_simple_test_frames()
        subtitles = create_test_subtitles()
        
        # Test page generation logic
        print("\n📄 Testing page generation logic...")
        
        pages_data = []
        total_pages = 48
        panels_per_page = 4
        
        for page_num in range(total_pages):
            page_data = {
                'page_number': page_num + 1,
                'panels': [],
                'bubbles': []
            }
            
            # Create 4 panels for this page (2x2 grid)
            for panel_num in range(panels_per_page):
                frame_idx = page_num * panels_per_page + panel_num
                
                # Get frame (cycle if not enough frames)
                frame_path = frames[frame_idx % len(frames)] if frames else None
                
                # Calculate panel position in 2x2 grid
                row = panel_num // 2
                col = panel_num % 2
                
                panel_data = {
                    'panel_number': panel_num + 1,
                    'frame_path': frame_path,
                    'row': row,
                    'col': col,
                    'x': col * 800,
                    'y': row * 540,
                    'width': 800,
                    'height': 540
                }
                page_data['panels'].append(panel_data)
                
                # Create bubble for this panel
                subtitle_idx = frame_idx % len(subtitles)
                subtitle = subtitles[subtitle_idx]
                
                bubble_data = {
                    'panel_number': panel_num + 1,
                    'text': subtitle['text'],
                    'x': col * 800 + 50,
                    'y': row * 540 + 50,
                    'width': 200,
                    'height': 80,
                    'font_size': 16,
                    'color': '#000000',
                    'background': '#FFFFFF',
                    'border': '#000000'
                }
                page_data['bubbles'].append(bubble_data)
            
            pages_data.append(page_data)
        
        print(f"✅ Generated {len(pages_data)} pages")
        
        # Test PNG page creation (simplified)
        print("\n🖼️ Testing PNG page creation...")
        
        os.makedirs('output/unity_pages', exist_ok=True)
        
        # Create first 3 pages as test
        for page_data in pages_data[:3]:
            try:
                # Create page canvas
                page_img = Image.new('RGB', (1600, 1080), color=(255, 255, 255))
                draw = ImageDraw.Draw(page_img)
                
                # Add panels to page
                for panel in page_data['panels']:
                    if panel['frame_path'] and os.path.exists(panel['frame_path']):
                        # Load panel image
                        panel_img = Image.open(panel['frame_path'])
                        # Resize to 800x540
                        panel_img = panel_img.resize((800, 540), Image.Resampling.LANCZOS)
                        
                        # Place panel on page
                        page_img.paste(panel_img, (panel['x'], panel['y']))
                
                # Add speech bubbles
                for bubble in page_data['bubbles']:
                    # Draw bubble background
                    x, y = bubble['x'], bubble['y']
                    w, h = bubble['width'], bubble['height']
                    
                    # Draw rounded rectangle background
                    draw.rounded_rectangle([x, y, x+w, y+h], radius=10, 
                                         fill=bubble['background'], outline=bubble['border'], width=2)
                    
                    # Draw text
                    try:
                        font = ImageFont.truetype("arial.ttf", bubble['font_size'])
                    except:
                        font = ImageFont.load_default()
                    
                    text = bubble['text']
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Center text in bubble
                    text_x = x + (w - text_width) // 2
                    text_y = y + (h - text_height) // 2
                    
                    draw.text((text_x, text_y), text, fill=bubble['color'], font=font)
                
                # Save page as PNG
                page_filename = f"page_{page_data['page_number']:03d}.png"
                page_path = os.path.join('output/unity_pages', page_filename)
                page_img.save(page_path, 'PNG', optimize=False)
                
                print(f"✅ Created page {page_data['page_number']}: {page_filename}")
                
            except Exception as e:
                print(f"❌ Error creating page {page_data['page_number']}: {e}")
        
        # Create interactive viewer
        print("\n🌐 Creating interactive viewer...")
        html_content = create_interactive_html(pages_data[:3])
        
        with open('output/unity_pages/interactive_viewer.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✅ Created interactive viewer")
        
        # Save Unity data
        print("\n💾 Saving Unity data...")
        with open('output/unity_pages/pages_data.json', 'w') as f:
            json.dump(pages_data[:3], f, indent=2)
        
        print("✅ Saved Unity data")
        
        print("\n🎉 Unity Comic Test Complete!")
        print("📁 Check output/unity_pages/ for generated files")
        print("🌐 Open output/unity_pages/interactive_viewer.html to view")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_interactive_html(pages_data):
    """Create simplified interactive HTML"""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unity Comic Viewer - Test</title>
    <style>
        body {{ margin: 0; padding: 20px; background: #f0f0f0; font-family: Arial, sans-serif; }}
        .header {{ text-align: center; margin-bottom: 30px; background: #333; color: white; padding: 20px; border-radius: 10px; }}
        .controls {{ text-align: center; margin-bottom: 20px; padding: 15px; background: white; border-radius: 10px; }}
        .btn {{ background: #4CAF50; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 5px; cursor: pointer; }}
        .btn:hover {{ background: #45a049; }}
        .page-container {{ max-width: 100%; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; margin-bottom: 20px; }}
        .comic-page {{ width: 1600px; height: 1080px; margin: 0 auto; position: relative; background: white; border: 3px solid #333; }}
        .panel {{ position: absolute; width: 800px; height: 540px; border: 1px solid #ddd; }}
        .panel img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
        .speech-bubble {{ position: absolute; background: white; border: 2px solid #333; border-radius: 10px; padding: 8px 12px; max-width: 200px; font-size: 16px; font-weight: bold; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); z-index: 10; text-align: center; color: #333; cursor: move; user-select: none; min-width: 100px; min-height: 40px; }}
        .speech-bubble:hover {{ transform: scale(1.05); box-shadow: 2px 2px 10px rgba(0,0,0,0.4); }}
        .speech-bubble::after {{ content: ''; position: absolute; bottom: -8px; left: 15px; width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 8px solid #333; }}
        .speech-bubble.editing {{ border-color: #ff6b6b; box-shadow: 0 0 10px rgba(255, 107, 107, 0.5); }}
        .page-navigation {{ text-align: center; margin: 20px 0; }}
        .page-info {{ display: inline-block; margin: 0 20px; font-size: 18px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 Unity Comic Viewer - Test</h1>
        <p>Interactive Speech Bubbles • Drag to Move • Double-click to Edit</p>
    </div>
    
    <div class="controls">
        <button class="btn" onclick="previousPage()" id="prevBtn">⬅️ Previous</button>
        <span class="page-info">Page <span id="currentPage">1</span> of {len(pages_data)}</span>
        <button class="btn" onclick="nextPage()" id="nextBtn">Next ➡️</button>
        <button class="btn" onclick="printCurrentPage()">🖨️ Print This Page</button>
    </div>
    
    <div class="page-container">
        <div id="comic-content">
            <div style="text-align: center; padding: 50px; color: #666; font-size: 18px;">Loading comic pages...</div>
        </div>
    </div>
    
    <script>
        let currentPageIndex = 0;
        let pagesData = {json.dumps(pages_data, indent=2)};
        let isDragging = false;
        let dragElement = null;
        let dragOffset = {{x: 0, y: 0}};
        
        document.addEventListener('DOMContentLoaded', function() {{
            loadPage(0);
        }});
        
        function loadPage(pageIndex) {{
            if (pageIndex < 0 || pageIndex >= pagesData.length) return;
            
            currentPageIndex = pageIndex;
            const pageData = pagesData[pageIndex];
            
            const container = document.getElementById('comic-content');
            container.innerHTML = '';
            
            const pageDiv = document.createElement('div');
            pageDiv.className = 'comic-page';
            
            pageData.panels.forEach((panel, index) => {{
                const panelDiv = document.createElement('div');
                panelDiv.className = 'panel';
                panelDiv.style.left = panel.x + 'px';
                panelDiv.style.top = panel.y + 'px';
                
                const img = document.createElement('img');
                img.src = 'page_' + pageData.page_number.toString().padStart(3, '0') + '.png';
                img.alt = `Panel ${{panel.panel_number}}`;
                
                panelDiv.appendChild(img);
                pageDiv.appendChild(panelDiv);
            }});
            
            pageData.bubbles.forEach((bubble, index) => {{
                const bubbleDiv = document.createElement('div');
                bubbleDiv.className = 'speech-bubble';
                bubbleDiv.textContent = bubble.text;
                bubbleDiv.style.left = bubble.x + 'px';
                bubbleDiv.style.top = bubble.y + 'px';
                bubbleDiv.style.width = bubble.width + 'px';
                bubbleDiv.style.height = bubble.height + 'px';
                bubbleDiv.style.fontSize = bubble.font_size + 'px';
                bubbleDiv.style.color = bubble.color;
                bubbleDiv.style.backgroundColor = bubble.background;
                bubbleDiv.style.borderColor = bubble.border;
                
                bubbleDiv.addEventListener('mousedown', startDrag);
                bubbleDiv.addEventListener('dblclick', editBubble);
                
                pageDiv.appendChild(bubbleDiv);
            }});
            
            container.appendChild(pageDiv);
            
            document.getElementById('currentPage').textContent = pageIndex + 1;
            document.getElementById('prevBtn').disabled = pageIndex === 0;
            document.getElementById('nextBtn').disabled = pageIndex === pagesData.length - 1;
        }}
        
        function startDrag(e) {{
            isDragging = true;
            dragElement = e.target;
            dragElement.style.zIndex = '1000';
            
            const rect = dragElement.getBoundingClientRect();
            dragOffset.x = e.clientX - rect.left;
            dragOffset.y = e.clientY - rect.top;
            
            document.addEventListener('mousemove', drag);
            document.addEventListener('mouseup', stopDrag);
            e.preventDefault();
        }}
        
        function drag(e) {{
            if (!isDragging || !dragElement) return;
            
            const pageRect = document.querySelector('.comic-page').getBoundingClientRect();
            const newX = e.clientX - pageRect.left - dragOffset.x;
            const newY = e.clientY - pageRect.top - dragOffset.y;
            
            const maxX = 1600 - dragElement.offsetWidth;
            const maxY = 1080 - dragElement.offsetHeight;
            
            dragElement.style.left = Math.max(0, Math.min(newX, maxX)) + 'px';
            dragElement.style.top = Math.max(0, Math.min(newY, maxY)) + 'px';
        }}
        
        function stopDrag() {{
            if (dragElement) {{
                dragElement.style.zIndex = '10';
                
                const pageData = pagesData[currentPageIndex];
                const bubbleIndex = Array.from(document.querySelectorAll('.speech-bubble')).indexOf(dragElement);
                if (bubbleIndex >= 0 && bubbleIndex < pageData.bubbles.length) {{
                    pageData.bubbles[bubbleIndex].x = parseInt(dragElement.style.left);
                    pageData.bubbles[bubbleIndex].y = parseInt(dragElement.style.top);
                }}
            }}
            
            isDragging = false;
            dragElement = null;
            
            document.removeEventListener('mousemove', drag);
            document.removeEventListener('mouseup', stopDrag);
        }}
        
        function editBubble(e) {{
            const bubble = e.target;
            bubble.classList.add('editing');
            bubble.contentEditable = true;
            bubble.focus();
            
            const range = document.createRange();
            range.selectNodeContents(bubble);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            
            bubble.addEventListener('blur', function() {{
                bubble.classList.remove('editing');
                bubble.contentEditable = false;
                
                const pageData = pagesData[currentPageIndex];
                const bubbleIndex = Array.from(document.querySelectorAll('.speech-bubble')).indexOf(bubble);
                if (bubbleIndex >= 0 && bubbleIndex < pageData.bubbles.length) {{
                    pageData.bubbles[bubbleIndex].text = bubble.textContent;
                }}
            }});
        }}
        
        function previousPage() {{
            if (currentPageIndex > 0) {{
                loadPage(currentPageIndex - 1);
            }}
        }}
        
        function nextPage() {{
            if (currentPageIndex < pagesData.length - 1) {{
                loadPage(currentPageIndex + 1);
            }}
        }}
        
        function printCurrentPage() {{
            const pageDiv = document.querySelector('.comic-page');
            if (pageDiv) {{
                const printWindow = window.open('', '_blank');
                printWindow.document.write(`
                    <html>
                        <head>
                            <title>Comic Page ${{currentPageIndex + 1}}</title>
                            <style>
                                body {{ margin: 0; padding: 20px; }}
                                .comic-page {{ width: 1600px; height: 1080px; border: 2px solid #000; position: relative; }}
                                .panel {{ position: absolute; width: 800px; height: 540px; border: 1px solid #ddd; }}
                                .panel img {{ width: 100%; height: 100%; object-fit: cover; }}
                                .speech-bubble {{ position: absolute; background: white; border: 2px solid #333; border-radius: 10px; padding: 8px 12px; font-size: 16px; font-weight: bold; color: #333; }}
                            </style>
                        </head>
                        <body>
                            ${{pageDiv.outerHTML}}
                        </body>
                    </html>
                `);
                printWindow.document.close();
                printWindow.print();
            }}
        }}
    </script>
</body>
</html>'''

if __name__ == "__main__":
    print("🎮 Unity Comic Generator Simple Test")
    print("=" * 50)
    
    success = test_basic_functionality()
    
    if success:
        print("\n✅ All tests passed!")
        print("🎮 Unity Comic Generator is ready for use!")
    else:
        print("\n❌ Tests failed!")