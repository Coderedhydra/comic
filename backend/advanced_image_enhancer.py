"""
Advanced Image Enhancement using State-of-the-Art AI Models
Real-ESRGAN, GFPGAN, and other cutting-edge models
Optimized for NVIDIA RTX 3050
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
import os
import requests
from io import BytesIO
import time
from typing import Optional, Tuple
from backend.ai_model_manager import get_ai_model_manager

class AdvancedImageEnhancer:
    """Advanced image enhancement using state-of-the-art AI models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Using device: {self.device}")
        
        # Initialize AI model manager
        self.ai_manager = get_ai_model_manager()
        
        # Enhancement settings
        self.use_ai_models = os.getenv('USE_AI_MODELS', '1') == '1'
        self.enhance_faces = os.getenv('ENHANCE_FACES', '1') == '1'
        self.use_anime_model = False  # Will be set based on content
        
        # Initialize models
        self._load_models()
        
    def _load_models(self):
        """Load advanced AI enhancement models"""
        try:
            print("🚀 Loading advanced AI models...")
            
            if self.use_ai_models:
                # Load Real-ESRGAN for super resolution
                self.ai_manager.load_realesrgan('RealESRGAN_x4plus')
                
                # Pre-load anime model for comic style
                self.ai_manager.load_realesrgan('RealESRGAN_x4plus_anime_6B')
                
                # Load GFPGAN for face enhancement
                if self.enhance_faces:
                    self.ai_manager.load_gfpgan()
                
                self.advanced_available = True
                print("✅ AI models loaded successfully")
            else:
                print("⚠️ AI models disabled, using traditional methods")
                self.advanced_available = False
            
        except Exception as e:
            print(f"⚠️ AI models failed to load: {e}")
            print("⚠️ Falling back to traditional enhancement methods")
            self.advanced_available = False
    
    def enhance_image(self, image_path: str, output_path: str = None) -> str:
        """Apply advanced image enhancement"""
        if output_path is None:
            output_path = image_path
        
        print(f"🚀 Enhancing image: {os.path.basename(image_path)}")
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Failed to load image: {image_path}")
                return image_path
            
            # Apply enhancement pipeline
            enhanced_img = self._apply_enhancement_pipeline(img)
            
            # Save enhanced image with maximum quality
            cv2.imwrite(output_path, enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            print(f"✅ Enhanced image saved: {os.path.basename(output_path)}")
            return output_path
            
        except Exception as e:
            print(f"❌ Enhancement failed: {e}")
            return image_path
    
    def _apply_enhancement_pipeline(self, img: np.ndarray) -> np.ndarray:
        """Apply complete enhancement pipeline with AI models"""
        original_img = img.copy()
        
        print("🎨 Applying AI-powered enhancement pipeline...")
        
        # Detect if image is anime/comic style
        self.use_anime_model = self._detect_anime_style(img)
        
        if self.advanced_available and self.use_ai_models:
            try:
                # 1. AI Super Resolution with Real-ESRGAN (4x upscaling)
                print("  🚀 Applying AI super resolution...")
                img = self.ai_manager.enhance_image_realesrgan(
                    img, 
                    use_anime_model=self.use_anime_model
                )
                
                # 2. AI Face Enhancement with GFPGAN
                if self.enhance_faces:
                    print("  👤 Enhancing faces with AI...")
                    img = self.ai_manager.enhance_face_gfpgan(img)
                
                # 3. Post-processing
                img = self.ai_manager.post_process(img)
                
                # Clear GPU memory for RTX 3050
                self.ai_manager.clear_memory()
                
                return img
                
            except Exception as e:
                print(f"⚠️ AI enhancement failed: {e}, using fallback")
                img = original_img
        
        # Fallback to traditional methods if AI models not available
        print("  📈 Using traditional enhancement methods...")
        
        # 1. Traditional Super Resolution
        img = self._apply_super_resolution_advanced(img)
        
        # 2. Advanced Color Enhancement
        img = self._enhance_colors_advanced(img)
        
        # 3. Advanced Noise Reduction
        img = self._reduce_noise_advanced(img)
        
        # 4. Advanced Sharpness Enhancement
        img = self._enhance_sharpness_advanced(img)
        
        # 5. Advanced Dynamic Range Optimization
        img = self._optimize_dynamic_range_advanced(img)
        
        # 6. Traditional Face Enhancement
        img = self._enhance_faces_advanced(img)
        
        return img
    
    def _apply_super_resolution_advanced(self, img: np.ndarray) -> np.ndarray:
        """Advanced super resolution (4x upscaling)"""
        try:
            print("📈 Applying advanced super resolution (4x upscaling)...")
            
            # Get original dimensions
            height, width = img.shape[:2]
            
            # Calculate target dimensions (4x upscaling)
            target_width = width * 4
            target_height = height * 4
            
            # Use LANCZOS interpolation for highest quality
            img = cv2.resize(img, (target_width, target_height), 
                           interpolation=cv2.INTER_LANCZOS4)
            
            # Apply additional sharpening after upscaling
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img = cv2.filter2D(img, -1, kernel)
            
            print(f"✅ Super resolution completed: {width}x{height} → {target_width}x{target_height}")
            
        except Exception as e:
            print(f"⚠️ Super resolution failed: {e}")
        
        return img
    
    def _enhance_colors_advanced(self, img: np.ndarray) -> np.ndarray:
        """Advanced color enhancement"""
        try:
            print("🎨 Applying advanced color enhancement...")
            
            # Convert to LAB color space for better color processing
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Enhance L channel (lightness) with CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            
            # Enhance A and B channels (color) with adaptive scaling
            lab[:,:,1] = cv2.convertScaleAbs(lab[:,:,1], alpha=1.3, beta=10)
            lab[:,:,2] = cv2.convertScaleAbs(lab[:,:,2], alpha=1.3, beta=10)
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Additional color saturation enhancement
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.convertScaleAbs(hsv[:,:,1], alpha=1.4, beta=0)  # Increase saturation
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        except Exception as e:
            print(f"⚠️ Color enhancement failed: {e}")
            enhanced = img
        
        return enhanced
    
    def _reduce_noise_advanced(self, img: np.ndarray) -> np.ndarray:
        """Advanced noise reduction"""
        try:
            print("🧹 Applying advanced noise reduction...")
            
            # Multi-stage noise reduction
            
            # 1. Bilateral filter for edge-preserving smoothing
            denoised = cv2.bilateralFilter(img, 9, 75, 75)
            
            # 2. Non-local means denoising for additional noise reduction
            denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)
            
            # 3. Gaussian blur for final smoothing
            denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
            
            # 4. Edge-preserving filter
            denoised = cv2.edgePreservingFilter(denoised, flags=1, sigma_s=60, sigma_r=0.4)
            
        except Exception as e:
            print(f"⚠️ Noise reduction failed: {e}")
            denoised = img
        
        return denoised
    
    def _enhance_sharpness_advanced(self, img: np.ndarray) -> np.ndarray:
        """Advanced sharpness enhancement"""
        try:
            print("🔪 Applying advanced sharpness enhancement...")
            
            # Multi-stage sharpening
            
            # 1. Unsharp masking
            gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
            sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
            
            # 2. Edge enhancement
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(sharpened, -1, kernel)
            
            # 3. Laplacian sharpening
            gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            sharpened = cv2.addWeighted(sharpened, 1.0, cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR), 0.3, 0)
            
        except Exception as e:
            print(f"⚠️ Sharpness enhancement failed: {e}")
            sharpened = img
        
        return sharpened
    
    def _optimize_dynamic_range_advanced(self, img: np.ndarray) -> np.ndarray:
        """Advanced dynamic range optimization"""
        try:
            print("📊 Applying advanced dynamic range optimization...")
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            
            # Enhance contrast in A and B channels
            lab[:,:,1] = cv2.convertScaleAbs(lab[:,:,1], alpha=1.2, beta=0)
            lab[:,:,2] = cv2.convertScaleAbs(lab[:,:,2], alpha=1.2, beta=0)
            
            # Convert back to BGR
            optimized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Additional contrast enhancement
            optimized = cv2.convertScaleAbs(optimized, alpha=1.1, beta=5)
            
        except Exception as e:
            print(f"⚠️ Dynamic range optimization failed: {e}")
            optimized = img
        
        return optimized
    
    def _enhance_faces_advanced(self, img: np.ndarray) -> np.ndarray:
        """Advanced face enhancement"""
        try:
            print("👤 Applying advanced face enhancement...")
            
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                print(f"🎭 Found {len(faces)} faces, applying enhancement...")
                
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_roi = img[y:y+h, x:x+w]
                    
                    # Apply face-specific enhancement
                    enhanced_face = self._enhance_face_region(face_roi)
                    
                    # Replace face region
                    img[y:y+h, x:x+w] = enhanced_face
            else:
                print("👤 No faces detected, skipping face enhancement")
                
        except Exception as e:
            print(f"⚠️ Face enhancement failed: {e}")
        
        return img
    
    def _enhance_face_region(self, face_img: np.ndarray) -> np.ndarray:
        """Enhance a specific face region"""
        try:
            # Apply gentle smoothing to face
            enhanced = cv2.bilateralFilter(face_img, 5, 50, 50)
            
            # Enhance skin tone
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.convertScaleAbs(hsv[:,:,1], alpha=1.1, beta=0)  # Gentle saturation boost
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Apply subtle sharpening
            kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
        except Exception as e:
            enhanced = face_img
        
        return enhanced
    
    def _detect_anime_style(self, img: np.ndarray) -> bool:
        """Detect if image is anime/manga/comic style"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Edge density check - anime has cleaner edges
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 2. Color count check - anime has fewer unique colors
            unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            
            # 3. Gradient smoothness - anime has smoother gradients
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            gradient_variance = np.var(laplacian)
            
            # Decision logic
            is_anime = (
                edge_density < 0.15 and  # Clean edges
                unique_colors < 10000 and  # Limited color palette
                gradient_variance < 1000  # Smooth gradients
            )
            
            if is_anime:
                print("  🎌 Detected anime/comic style - using specialized model")
            
            return is_anime
            
        except Exception as e:
            print(f"⚠️ Style detection failed: {e}")
            return False
    
    def enhance_batch(self, image_paths: list, output_dir: str = None) -> list:
        """Enhance multiple images"""
        if output_dir is None:
            output_dir = "enhanced"
        
        os.makedirs(output_dir, exist_ok=True)
        enhanced_paths = []
        
        print(f"🎯 Enhancing {len(image_paths)} images with advanced techniques...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"📸 Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Generate output path
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"enhanced_{filename}")
            
            # Enhance image
            enhanced_path = self.enhance_image(image_path, output_path)
            enhanced_paths.append(enhanced_path)
        
        print(f"✅ Enhanced {len(enhanced_paths)} images with advanced techniques")
        return enhanced_paths

# Global instance
advanced_enhancer = None

def get_advanced_enhancer():
    """Get or create global advanced enhancer instance"""
    global advanced_enhancer
    if advanced_enhancer is None:
        advanced_enhancer = AdvancedImageEnhancer()
    return advanced_enhancer

if __name__ == "__main__":
    # Test the enhancer
    enhancer = AdvancedImageEnhancer()
    print("🧪 Advanced Image Enhancer ready for testing!")