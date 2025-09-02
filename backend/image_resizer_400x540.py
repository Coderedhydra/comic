"""
Resize images to exactly 400x540 for comic panels
"""

import os
import json
from typing import List, Tuple

class ImageResizer400x540:
    """Resize images to exact 400x540 dimensions"""
    
    def __init__(self):
        self.target_width = 400
        self.target_height = 540
        
    def get_resize_command(self, input_path: str, output_path: str, fit_mode: str = "contain") -> str:
        """
        Generate ffmpeg command to resize image to 400x540
        
        fit_mode options:
        - "contain": Fit entire image, add padding if needed (no zoom)
        - "cover": Fill entire area, crop if needed (may zoom)
        - "stretch": Stretch to exact size (may distort)
        """
        
        if fit_mode == "contain":
            # Fit image without cropping, add padding
            filter_cmd = (
                f"scale={self.target_width}:{self.target_height}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={self.target_width}:{self.target_height}:"
                f"(ow-iw)/2:(oh-ih)/2:black"
            )
        elif fit_mode == "cover":
            # Fill area, crop excess
            filter_cmd = (
                f"scale={self.target_width}:{self.target_height}:"
                f"force_original_aspect_ratio=increase,"
                f"crop={self.target_width}:{self.target_height}"
            )
        else:  # stretch
            # Stretch to exact size
            filter_cmd = f"scale={self.target_width}:{self.target_height}"
        
        return f'ffmpeg -i "{input_path}" -vf "{filter_cmd}" -y "{output_path}"'
    
    def process_frames_batch(self, frames_dir: str, output_dir: str, fit_mode: str = "contain") -> List[str]:
        """Process all frames in directory to 400x540"""
        
        os.makedirs(output_dir, exist_ok=True)
        processed_files = []
        
        # Get all PNG files
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        
        print(f"📐 Resizing {len(frame_files)} frames to 400x540...")
        
        for i, frame_file in enumerate(frame_files):
            input_path = os.path.join(frames_dir, frame_file)
            output_path = os.path.join(output_dir, f"panel_{i+1:03d}.png")
            
            # Generate resize command
            cmd = self.get_resize_command(input_path, output_path, fit_mode)
            
            # Execute command
            result = os.system(cmd)
            
            if result == 0:
                processed_files.append(output_path)
                print(f"  ✓ Resized {frame_file} -> panel_{i+1:03d}.png")
            else:
                print(f"  ✗ Failed to resize {frame_file}")
        
        print(f"✅ Resized {len(processed_files)} frames to 400x540")
        return processed_files
    
    def create_resize_script(self, frames_dir: str, output_dir: str) -> str:
        """Create a shell script to resize all images"""
        
        script_content = f"""#!/bin/bash
# Resize all frames to 400x540 for comic panels

FRAMES_DIR="{frames_dir}"
OUTPUT_DIR="{output_dir}"

mkdir -p "$OUTPUT_DIR"

echo "📐 Resizing frames to 400x540..."

# Process each PNG file
for file in "$FRAMES_DIR"/*.png; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        base_name="${{filename%.*}}"
        
        # Resize to 400x540 with padding (no zoom/crop)
        ffmpeg -i "$file" \\
            -vf "scale=400:540:force_original_aspect_ratio=decrease,pad=400:540:(ow-iw)/2:(oh-ih)/2:black" \\
            -y "$OUTPUT_DIR/${{base_name}}_400x540.png"
        
        echo "✓ Resized $filename"
    fi
done

echo "✅ Resize complete! Check $OUTPUT_DIR"
"""
        
        script_path = "resize_panels_400x540.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        print(f"📄 Created resize script: {script_path}")
        
        return script_path

# Utility functions
def resize_for_exact_layout(frames_dir: str = "frames/final", output_dir: str = "frames/panels_400x540"):
    """Resize all frames to exactly 400x540 for perfect 800x1080 layout"""
    
    resizer = ImageResizer400x540()
    
    # Create resize script
    script_path = resizer.create_resize_script(frames_dir, output_dir)
    
    print("\n📋 To resize your images to 400x540:")
    print(f"1. Run: bash {script_path}")
    print(f"2. Resized images will be in: {output_dir}/")
    print("3. These will fit perfectly in the 800x1080 layout without zooming")
    
    return script_path