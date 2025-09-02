"""
Generate image files for each comic page at 800x1080 resolution
Simple version that creates HTML canvases instead of actual image files
"""

import os
import json
from typing import List, Dict

class PageImageGenerator:
    """Generate page images as HTML canvases that can be saved"""
    
    def __init__(self, output_dir: str = "output/page_images"):
        self.output_dir = output_dir
        self.page_size = (800, 1080)  # Width x Height
        
    def generate_page_images(self, pages_data: List[Dict], frames_dir: str) -> List[str]:
        """Generate HTML pages that render as images"""
        os.makedirs(self.output_dir, exist_ok=True)
        generated_files = []
        
        # Create individual HTML files for each page
        for i, page in enumerate(pages_data):
            filename = f"page_{i+1:03d}.html"
            filepath = os.path.join(self.output_dir, filename)
            self._create_page_html(page, frames_dir, i + 1, filepath)
            generated_files.append(filepath)
            print(f"📄 Generated page {i+1}/{len(pages_data)}: {filename}")
        
        # Create main gallery
        self._create_gallery_html(len(pages_data))
        
        return generated_files
    
    def _create_page_html(self, page: Dict, frames_dir: str, page_num: int, output_path: str):
        """Create HTML that renders a comic page at 800x1080"""
        
        panels_html = ""
        panels = page.get('panels', [])
        
        for idx, panel in enumerate(panels[:4]):
            if panel.get('image'):
                img_path = f"../../frames/final/{panel['image']}"
                bubble_html = ""
                
                if panel.get('speech_bubble'):
                    bubble = panel['speech_bubble']
                    bubble_html = f"""
                    <div class="speech-bubble" style="left: {bubble.get('x', 50)}%; top: {bubble.get('y', 20)}%;">
                        {bubble.get('text', '')}
                    </div>
                    """
                
                panels_html += f"""
                <div class="panel panel-{idx+1}">
                    <img src="{img_path}" alt="Panel {idx+1}">
                    {bubble_html}
                </div>
                """
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comic Page {page_num}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #f0f0f0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
        
        .page-container {{
            width: 800px;
            height: 1080px;
            background: white;
            position: relative;
            box-shadow: 0 0 20px rgba(0,0,0,0.3);
        }}
        
        .comic-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 10px;
            padding: 20px;
            height: calc(100% - 60px);
        }}
        
        .panel {{
            border: 3px solid black;
            overflow: hidden;
            position: relative;
            background: white;
        }}
        
        .panel img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        
        .speech-bubble {{
            position: absolute;
            background: white;
            border: 2px solid black;
            border-radius: 15px;
            padding: 10px 15px;
            max-width: 60%;
            transform: translate(-50%, -50%);
            text-align: center;
            font-family: Arial, sans-serif;
            font-size: 14px;
            font-weight: bold;
        }}
        
        .page-number {{
            text-align: center;
            padding: 10px;
            color: #666;
            font-family: Arial, sans-serif;
        }}
        
        .download-btn {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        }}
        
        .download-btn:hover {{
            background: #45a049;
        }}
        
        @media print {{
            body {{
                margin: 0;
                background: white;
            }}
            .download-btn {{
                display: none;
            }}
            .page-container {{
                box-shadow: none;
                width: 100%;
                height: 100vh;
                max-width: 800px;
                max-height: 1080px;
                margin: 0 auto;
            }}
        }}
    </style>
</head>
<body>
    <div class="page-container" id="comic-page">
        <div class="comic-grid">
            {panels_html}
        </div>
        <div class="page-number">Page {page_num}</div>
    </div>
    
    <button class="download-btn" onclick="downloadAsImage()">
        📥 Download as Image
    </button>
    
    <script>
        function downloadAsImage() {{
            // Use browser print to PDF as image alternative
            window.print();
        }}
        
        // Auto-size to fit screen while maintaining aspect ratio
        function resizePage() {{
            const container = document.querySelector('.page-container');
            const maxWidth = window.innerWidth - 40;
            const maxHeight = window.innerHeight - 40;
            const scale = Math.min(maxWidth / 800, maxHeight / 1080, 1);
            
            if (scale < 1) {{
                container.style.transform = `scale(${{scale}})`;
                container.style.transformOrigin = 'center center';
            }}
        }}
        
        window.addEventListener('resize', resizePage);
        resizePage();
    </script>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _create_gallery_html(self, num_pages: int):
        """Create gallery index HTML"""
        
        page_links = ""
        for i in range(num_pages):
            page_num = i + 1
            filename = f"page_{page_num:03d}.html"
            page_links += f"""
            <div class="page-card">
                <a href="{filename}" target="_blank">
                    <div class="page-preview">
                        <div class="page-number-large">{page_num}</div>
                        <div class="page-label">Page {page_num}</div>
                    </div>
                </a>
                <div class="page-actions">
                    <a href="{filename}" target="_blank">🔍 View</a>
                    <a href="{filename}" download>📥 Download HTML</a>
                </div>
            </div>
            """
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comic Page Images Gallery</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f0f0f0;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        
        .header p {{
            color: #666;
            margin: 5px 0;
        }}
        
        .instructions {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            text-align: left;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }}
        
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .page-card {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .page-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        
        .page-preview {{
            height: 270px;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }}
        
        .page-preview::before {{
            content: "";
            position: absolute;
            inset: 10px;
            border: 3px solid #ddd;
            border-radius: 5px;
        }}
        
        .page-number-large {{
            font-size: 48px;
            font-weight: bold;
            color: #666;
            margin-bottom: 10px;
        }}
        
        .page-label {{
            color: #888;
            font-size: 14px;
        }}
        
        .page-actions {{
            padding: 10px;
            text-align: center;
            background: #fafafa;
            border-top: 1px solid #eee;
        }}
        
        .page-actions a {{
            margin: 0 5px;
            color: #2196F3;
            text-decoration: none;
            font-size: 14px;
        }}
        
        .page-actions a:hover {{
            text-decoration: underline;
        }}
        
        .export-section {{
            text-align: center;
            margin: 30px 0;
        }}
        
        .export-btn {{
            display: inline-block;
            padding: 12px 24px;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin: 0 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        
        .export-btn:hover {{
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 Comic Page Images</h1>
        <p>All pages rendered at 800x1080 resolution</p>
        <p>{num_pages} pages generated</p>
        
        <div class="instructions">
            <strong>💡 How to save as images:</strong>
            <ol>
                <li>Click on any page to view it</li>
                <li>Click "Download as Image" button</li>
                <li>In print dialog: Select "Save as PDF"</li>
                <li>Or take a screenshot (better quality)</li>
            </ol>
        </div>
        
        <div class="export-section">
            <a href="#" class="export-btn" onclick="openAll(); return false;">
                📂 Open All Pages
            </a>
        </div>
    </div>
    
    <div class="gallery">
        {page_links}
    </div>
    
    <script>
        function openAll() {{
            if (confirm('This will open {num_pages} new tabs. Continue?')) {{
                for (let i = 1; i <= {num_pages}; i++) {{
                    const filename = `page_${{String(i).padStart(3, '0')}}.html`;
                    window.open(filename, '_blank');
                }}
            }}
        }}
    </script>
</body>
</html>
"""
        
        index_path = os.path.join(self.output_dir, 'index.html')
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📋 Page gallery created: {index_path}")

def generate_page_images_from_json(json_path: str, frames_dir: str, output_dir: str = None):
    """Standalone function to generate page images from pages.json"""
    if not os.path.exists(json_path):
        print(f"❌ Pages JSON not found: {json_path}")
        return []
    
    # Load pages data
    with open(json_path, 'r') as f:
        pages_data = json.load(f)
    
    # Create generator
    generator = PageImageGenerator(output_dir or "output/page_images")
    
    # Generate images
    return generator.generate_page_images(pages_data, frames_dir)