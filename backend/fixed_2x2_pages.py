"""
Generate 12 meaningful panels in 2x2 grid format (3 pages × 4 panels each)
"""

from backend.class_def import panel, Page

def generate_12_panels_2x2_grid(frame_files, bubbles):
    """Generate 12 panels across 3 pages, each with 2x2 grid"""
    
    pages = []
    num_frames = min(12, len(frame_files))  # Max 12 panels
    panels_per_page = 4  # 2x2 grid = 4 panels
    
    print(f"📄 Generating {num_frames} panels in 2x2 grid format")
    print(f"📑 Creating {(num_frames + 3) // 4} pages")
    
    frame_idx = 0
    bubble_idx = 0
    
    # Create pages with 2x2 grid
    while frame_idx < num_frames:
        page_panels = []
        page_bubbles = []
        
        # Create 4 panels for this page (2x2 grid)
        for i in range(panels_per_page):
            if frame_idx < num_frames:
                # Each panel takes 1/4 of the page
                panel_obj = panel(
                    image=frame_files[frame_idx],
                    row_span=6,  # Half the height (12/2 = 6)
                    col_span=6   # Half the width (12/2 = 6)
                )
                page_panels.append(panel_obj)
                
                # Add corresponding bubble if available
                if bubble_idx < len(bubbles):
                    page_bubbles.append(bubbles[bubble_idx])
                    bubble_idx += 1
                
                frame_idx += 1
        
        # Create 2x2 arrangement
        arrangement = ['0101', '0101', '2323', '2323']  # 2x2 grid pattern
        
        # Create page
        page = Page(
            panels=page_panels,
            bubbles=page_bubbles,
            panel_arrangement=arrangement
        )
        pages.append(page)
        
        print(f"  ✓ Page {len(pages)}: {len(page_panels)} panels")
    
    return pages

def extract_12_meaningful_frames(all_frames, all_bubbles):
    """Extract only the 12 most meaningful frames from all available"""
    
    if len(all_frames) <= 12:
        return all_frames, all_bubbles
    
    print(f"🎯 Selecting 12 most meaningful frames from {len(all_frames)} total")
    
    # Simple selection: take evenly distributed frames
    # In real implementation, this would use story analysis
    step = len(all_frames) / 12
    selected_frames = []
    selected_bubbles = []
    
    for i in range(12):
        idx = int(i * step)
        selected_frames.append(all_frames[idx])
        if idx < len(all_bubbles):
            selected_bubbles.append(all_bubbles[idx])
    
    return selected_frames, selected_bubbles