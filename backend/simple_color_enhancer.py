"""
Simple color-preserving enhancement without AI models
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance

class SimpleColorEnhancer:
    """Simple enhancement that preserves colors"""
    
    def enhance_batch(self, frames_dir: str):
        """Enhance all frames in directory"""
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        
        print(f"🎨 Enhancing {len(frame_files)} frames with color preservation...")
        
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            
            # Check if this is the last frame
            is_last = (i == len(frame_files) - 1)
            if is_last:
                print(f"  ⚠️ Processing last frame {frame_file} with extra care...")
            
            self.enhance_single(frame_path, frame_path, skip_if_last=is_last)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(frame_files)} frames")
        
        print("✅ Color enhancement complete")
    
    def enhance_single(self, input_path: str, output_path: str, skip_if_last: bool = False):
        """Enhance single image with color preservation"""
        try:
            # Read image
            img = cv2.imread(input_path)
            if img is None:
                return
            
            # Skip enhancement for last frame if it has issues
            if skip_if_last:
                print(f"    Skipping enhancement for last frame to avoid color issues")
                # Just copy the file without changes
                return
            
            # 1. Denoise while preserving edges
            img = cv2.bilateralFilter(img, 5, 30, 30)
            
            # 2. Convert to PIL for easier color manipulation
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # 3. Very mild enhancement only
            # Brightness - SKIP or very minimal (images already bright)
            # brightness = ImageEnhance.Brightness(img_pil)
            # img_pil = brightness.enhance(1.0)  # No brightness change
            
            # Contrast - very subtle
            contrast = ImageEnhance.Contrast(img_pil)
            img_pil = contrast.enhance(1.05)  # Only 5% more contrast
            
            # Color - subtle boost
            color = ImageEnhance.Color(img_pil)
            img_pil = color.enhance(1.05)  # Only 5% more vibrant
            
            # 4. Convert back to OpenCV
            img_enhanced = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # 5. Very mild sharpening (reduced intensity)
            kernel = np.array([[0, -0.25, 0],
                              [-0.25, 2, -0.25],
                              [0, -0.25, 0]], dtype=np.float32)
            img_enhanced = cv2.filter2D(img_enhanced, -1, kernel)
            
            # 6. Ensure we don't clip colors
            img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)
            
            # Save with high quality
            cv2.imwrite(output_path, img_enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            
        except Exception as e:
            print(f"⚠️ Enhancement failed for {os.path.basename(input_path)}: {e}")