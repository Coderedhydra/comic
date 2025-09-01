"""
Quality and Color Enhancer - Improves image quality while preserving natural colors
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance

class QualityColorEnhancer:
    def __init__(self):
        self.enhance_quality = True
        self.enhance_colors = True
        
    def enhance_frame(self, frame_path: str, output_path: str = None) -> str:
        """Enhance frame quality and colors"""
        
        if output_path is None:
            output_path = frame_path
            
        try:
            # Read image
            img = cv2.imread(frame_path)
            if img is None:
                return frame_path
                
            print(f"🎨 Enhancing {frame_path}...")
            
            # 1. Denoise
            img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
            
            # 2. Improve sharpness
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            sharpened = cv2.filter2D(img, -1, kernel)
            
            # Blend with original (to avoid over-sharpening)
            img = cv2.addWeighted(img, 0.5, sharpened, 0.5, 0)
            
            # 3. Enhance colors using PIL
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Increase color vibrancy
            color_enhancer = ImageEnhance.Color(img_pil)
            img_pil = color_enhancer.enhance(1.3)  # 30% more colorful
            
            # Adjust brightness
            brightness_enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = brightness_enhancer.enhance(1.1)  # 10% brighter
            
            # Adjust contrast
            contrast_enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = contrast_enhancer.enhance(1.2)  # 20% more contrast
            
            # Convert back to OpenCV
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # 4. Auto white balance
            img = self._auto_white_balance(img)
            
            # 5. Enhance details in dark areas
            img = self._enhance_dark_areas(img)
            
            # Save with high quality
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            return output_path
            
        except Exception as e:
            print(f"⚠️ Enhancement failed: {e}")
            return frame_path
    
    def _auto_white_balance(self, img):
        """Simple auto white balance"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    def _enhance_dark_areas(self, img):
        """Enhance details in dark areas without affecting bright areas"""
        # Create a mask for dark areas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 100)  # Dark areas
        
        # Apply CLAHE only to dark areas
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        
        # Apply enhancement only to masked areas
        l_channel = np.where(mask > 0, enhanced_l, l_channel)
        lab[:, :, 0] = l_channel
        
        # Convert back
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return result
    
    def batch_enhance(self, frames_dir: str):
        """Enhance all frames in directory"""
        import os
        
        frames = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        print(f"🎨 Enhancing {len(frames)} frames for better quality and colors...")
        
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, frame)
            self.enhance_frame(frame_path)
            print(f"  ✓ Enhanced {i+1}/{len(frames)}")
        
        print("✅ Quality and color enhancement complete!")