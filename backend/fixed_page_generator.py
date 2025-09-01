"""
Fixed page generator that creates proper 12-panel comics
"""

from backend.class_def import panel, Page

def generate_12_panel_pages(frame_files, bubbles):
    """Generate pages with 12 panels in 3x4 grid"""
    
    pages = []
    num_frames = min(12, len(frame_files))  # Max 12 panels
    
    print(f"📄 Generating comic with {num_frames} panels in 3x4 grid")
    
    # Create single page with all panels
    panels = []
    
    # 3x4 grid = 3 rows, 4 columns
    # Each panel: row_span=4, col_span=3 (total 12x12 grid)
    for i in range(num_frames):
        panel_obj = panel(
            image=frame_files[i],
            row_span=4,  # 12/3 = 4
            col_span=3   # 12/4 = 3
        )
        panels.append(panel_obj)
    
    # Create arrangement string for 3x4 grid
    arrangement = []
    panel_idx = 0
    for row in range(3):
        row_str = ""
        for col in range(4):
            if panel_idx < num_frames:
                row_str += str(panel_idx % 10)
                panel_idx += 1
            else:
                row_str += "0"  # Empty space
        arrangement.append(row_str)
    
    # Get corresponding bubbles
    page_bubbles = bubbles[:num_frames] if bubbles else []
    
    # Create page
    page = Page(
        panels=panels,
        bubbles=page_bubbles,
        panel_arrangement=arrangement
    )
    pages.append(page)
    
    return pages

def generate_proper_layout(num_panels):
    """Generate proper layout configuration based on panel count"""
    
    if num_panels <= 6:
        return {
            'pages': 1,
            'panels_per_page': num_panels,
            'rows': 2,
            'cols': 3,
            'row_span': 6,
            'col_span': 4
        }
    elif num_panels <= 9:
        return {
            'pages': 1,
            'panels_per_page': num_panels,
            'rows': 3,
            'cols': 3,
            'row_span': 4,
            'col_span': 4
        }
    elif num_panels <= 12:
        return {
            'pages': 1,
            'panels_per_page': num_panels,
            'rows': 3,
            'cols': 4,
            'row_span': 4,
            'col_span': 3
        }
    else:
        # More than 12 panels - use multiple pages
        return {
            'pages': 2,
            'panels_per_page': 8,
            'rows': 2,
            'cols': 4,
            'row_span': 6,
            'col_span': 3
        }