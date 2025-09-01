"""
Generate 12 pages of 2x2 grid comics (48 panels total)
Only selecting meaningful story moments
"""

from backend.class_def import panel, Page

def generate_12_pages_2x2_grid(frame_files, bubbles):
    """Generate 12 pages, each with 2x2 grid (4 panels per page)"""
    
    pages = []
    panels_per_page = 4  # 2x2 grid
    total_pages = 12
    target_panels = total_pages * panels_per_page  # 48 panels
    
    # Select meaningful frames (up to 48)
    selected_frames = select_meaningful_frames(frame_files, target_panels)
    num_frames = len(selected_frames)
    
    print(f"📄 Generating {total_pages} pages with 2x2 grid")
    print(f"🎯 Selected {num_frames} meaningful frames from {len(frame_files)} total")
    
    frame_idx = 0
    bubble_idx = 0
    
    # Create 12 pages
    for page_num in range(total_pages):
        page_panels = []
        page_bubbles = []
        
        # Create 4 panels for this page (2x2 grid)
        for i in range(panels_per_page):
            if frame_idx < num_frames:
                # Each panel takes 1/4 of the page
                panel_obj = panel(
                    image=selected_frames[frame_idx],
                    row_span=6,  # Half the height (12/2 = 6)
                    col_span=6   # Half the width (12/2 = 6)
                )
                page_panels.append(panel_obj)
                
                # Add corresponding bubble if available
                if bubble_idx < len(bubbles):
                    page_bubbles.append(bubbles[bubble_idx])
                    bubble_idx += 1
                
                frame_idx += 1
            else:
                # Create empty panel if we run out of frames
                panel_obj = panel(
                    image=selected_frames[0] if selected_frames else 'blank.png',
                    row_span=6,
                    col_span=6
                )
                page_panels.append(panel_obj)
        
        # Create 2x2 arrangement
        # Panel layout:
        # [0][1]
        # [2][3]
        arrangement = [
            '0011',  # Row 1: Panel 0, Panel 1
            '0011',  # Row 2: Panel 0, Panel 1
            '0011',  # Row 3: Panel 0, Panel 1
            '2233',  # Row 4: Panel 2, Panel 3
            '2233',  # Row 5: Panel 2, Panel 3
            '2233'   # Row 6: Panel 2, Panel 3
        ]
        
        # Create page (Page class doesn't take panel_arrangement parameter)
        page = Page(
            panels=page_panels,
            bubbles=page_bubbles
        )
        pages.append(page)
        
        # Show progress
        panels_on_page = min(4, num_frames - (page_num * 4))
        if panels_on_page > 0:
            print(f"  ✓ Page {page_num + 1}: {panels_on_page} panels")
    
    print(f"✅ Generated {len(pages)} pages with {min(num_frames, target_panels)} total panels")
    return pages

def select_meaningful_frames(all_frames, target_count):
    """Select frames to tell complete story"""
    
    if len(all_frames) <= target_count:
        print(f"📚 Using all {len(all_frames)} frames (complete story)")
        return all_frames
    
    print(f"📚 Selecting {target_count} frames from {len(all_frames)} to tell complete story")
    
    # Smart selection based on story phases
    # Allocate frames to different story parts:
    # - Introduction: 8 panels (2 pages)
    # - Development: 16 panels (4 pages)
    # - Climax: 16 panels (4 pages)
    # - Resolution: 8 panels (2 pages)
    
    selected = []
    total = len(all_frames)
    
    # Introduction (first 15% of frames)
    intro_end = int(total * 0.15)
    intro_frames = all_frames[:intro_end]
    intro_step = max(1, len(intro_frames) // 8)
    selected.extend(intro_frames[::intro_step][:8])
    
    # Development (15% to 50%)
    dev_start = intro_end
    dev_end = int(total * 0.5)
    dev_frames = all_frames[dev_start:dev_end]
    dev_step = max(1, len(dev_frames) // 16)
    selected.extend(dev_frames[::dev_step][:16])
    
    # Climax (50% to 85%)
    climax_start = dev_end
    climax_end = int(total * 0.85)
    climax_frames = all_frames[climax_start:climax_end]
    climax_step = max(1, len(climax_frames) // 16)
    selected.extend(climax_frames[::climax_step][:16])
    
    # Resolution (last 15%)
    resolution_frames = all_frames[climax_end:]
    resolution_step = max(1, len(resolution_frames) // 8)
    selected.extend(resolution_frames[::resolution_step][:8])
    
    # Ensure we have exactly the target count
    if len(selected) > target_count:
        selected = selected[:target_count]
    elif len(selected) < target_count:
        # Fill with evenly distributed frames
        remaining = target_count - len(selected)
        step = total // remaining
        for i in range(remaining):
            idx = i * step
            if all_frames[idx] not in selected:
                selected.append(all_frames[idx])
    
    return selected[:target_count]