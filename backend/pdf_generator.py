"""
PDF Generator for Edited Comics
Generates PDF from comic with user edits preserved
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw, ImageFont
import json

class ComicPDFGenerator:
    """Generate PDF from edited comic data"""
    
    def __init__(self):
        self.page_width = A4[0]
        self.page_height = A4[1]
        self.margin = 20
        
    def generate_pdf(self, pages_data, edited_bubbles, output_path="output/comic_edited.pdf"):
        """
        Generate PDF with edited text and positions
        
        Args:
            pages_data: Original comic pages data
            edited_bubbles: List of edited bubble data (text, position)
            output_path: Output PDF path
        """
        
        # Create PDF
        c = canvas.Canvas(output_path, pagesize=A4)
        
        bubble_index = 0
        
        for page_num, page in enumerate(pages_data):
            if page_num > 0:
                c.showPage()
            
            # Add page title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(self.margin, self.page_height - 30, f"Page {page_num + 1}")
            
            # Calculate panel layout
            panels_per_page = len(page['panels'])
            if panels_per_page <= 4:
                cols, rows = 2, 2
            else:
                cols, rows = 3, 2
            
            panel_width = (self.page_width - self.margin * 2 - 10 * (cols - 1)) / cols
            panel_height = (self.page_height - 100 - 10 * (rows - 1)) / rows
            
            # Draw panels
            for i, panel in enumerate(page['panels']):
                row = i // cols
                col = i % cols
                
                x = self.margin + col * (panel_width + 10)
                y = self.page_height - 60 - (row + 1) * (panel_height + 10)
                
                # Draw panel image
                img_path = os.path.join('frames/final', panel['image'])
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img_reader = ImageReader(img)
                    c.drawImage(img_reader, x, y, width=panel_width, height=panel_height, preserveAspectRatio=True)
                
                # Draw border
                c.setStrokeColorRGB(0, 0, 0)
                c.setLineWidth(2)
                c.rect(x, y, panel_width, panel_height)
                
                # Draw bubbles for this panel
                for bubble in page.get('bubbles', []):
                    if bubble_index < len(edited_bubbles):
                        edited_bubble = edited_bubbles[bubble_index]
                        
                        # Use edited text and position
                        text = edited_bubble.get('text', bubble['dialog'])
                        
                        # Calculate bubble position
                        if edited_bubble.get('left') and edited_bubble.get('top'):
                            # Convert CSS position to PDF coordinates
                            bubble_x = x + self._parse_position(edited_bubble['left'], panel_width)
                            bubble_y = y + panel_height - self._parse_position(edited_bubble['top'], panel_height) - 30
                        else:
                            # Use default position
                            bubble_x = x + bubble['bubble_offset_x']
                            bubble_y = y + panel_height - bubble['bubble_offset_y'] - 30
                        
                        # Draw speech bubble
                        self._draw_speech_bubble(c, bubble_x, bubble_y, text)
                        
                    bubble_index += 1
        
        # Save PDF
        c.save()
        return output_path
    
    def _parse_position(self, css_value, max_value):
        """Convert CSS position (e.g., '50px') to numeric value"""
        if isinstance(css_value, str) and css_value.endswith('px'):
            return float(css_value[:-2])
        return 0
    
    def _draw_speech_bubble(self, canvas, x, y, text):
        """Draw a speech bubble with text"""
        # Measure text
        canvas.setFont("Helvetica-Bold", 10)
        text_width = canvas.stringWidth(text, "Helvetica-Bold", 10)
        
        # Wrap text if needed
        words = text.split()
        lines = []
        current_line = []
        max_width = 150
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if canvas.stringWidth(test_line, "Helvetica-Bold", 10) > max_width:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Calculate bubble size
        bubble_width = min(max_width + 20, 180)
        bubble_height = len(lines) * 15 + 20
        
        # Draw bubble background
        canvas.setFillColorRGB(1, 1, 1)
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.setLineWidth(2)
        
        # Rounded rectangle for bubble
        canvas.roundRect(x, y, bubble_width, bubble_height, 10, fill=1, stroke=1)
        
        # Draw text
        canvas.setFillColorRGB(0, 0, 0)
        text_y = y + bubble_height - 15
        for line in lines:
            canvas.drawString(x + 10, text_y, line)
            text_y -= 15
    
    def generate_from_html(self, html_path, edited_data, output_path="output/comic_edited.pdf"):
        """
        Alternative: Generate PDF from edited HTML
        This would require parsing the HTML and extracting positions
        """
        # This is a placeholder for HTML-based PDF generation
        # In practice, you might use tools like wkhtmltopdf or Playwright
        pass


def generate_edited_pdf(request_data):
    """
    Generate PDF from edit request
    
    Args:
        request_data: Dict with edited bubble data
    """
    generator = ComicPDFGenerator()
    
    # Load original pages data
    with open('output/pages.json', 'r') as f:
        pages_data = json.load(f)
    
    # Get edited bubbles
    edited_bubbles = request_data.get('bubbles', [])
    
    # Generate PDF
    output_path = generator.generate_pdf(pages_data, edited_bubbles)
    
    return output_path