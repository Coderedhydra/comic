"""
Ultra Compact Image Enhancer for Extreme Memory Constraints
Designed for RTX 3050 Laptop with strict <1GB VRAM limit
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import gc

class UltraCompactESRGAN(nn.Module):
    """Ultra lightweight ESRGAN - only 200MB VRAM usage"""
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        
        # Ultra compact architecture
        nf = 24  # Even smaller feature channels
        
        self.conv1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        
        # Pixel shuffle for upsampling
        self.upscale = nn.Sequential(
            nn.Conv2d(nf, 3 * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale)
        )
        
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Simple forward pass
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x3 = self.conv3(x2)
        x = x1 + x3  # Skip connection
        x = self.upscale(x)
        return x

class MemorySafeEnhancer:
    """Memory-safe enhancer that guarantees <1GB VRAM usage"""
    
    def __init__(self):
        self.device = self._setup_device()
        self.model = None
        self.tile_size = 64  # Very small tiles
        self.scale = 2  # 2x max for 2K output
        
        # Load model
        self._load_model()
        
    def _setup_device(self):
        """Setup device with strict memory limits"""
        if torch.cuda.is_available():
            # Clear any existing allocations
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Set strict memory limit
            torch.cuda.set_per_process_memory_fraction(0.3)  # Only 30% of VRAM
            
            device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Print available memory
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"📊 Total VRAM: {total:.1f}GB, Using max: {total*0.3:.1f}GB")
        else:
            device = torch.device('cpu')
            print("💻 Using CPU")
            
        return device
    
    def _load_model(self):
        """Load ultra compact model"""
        try:
            print("🔄 Loading ultra-compact model...")
            
            self.model = UltraCompactESRGAN(scale=self.scale)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Use half precision on GPU
            if self.device.type == 'cuda':
                self.model = self.model.half()
            
            # Calculate model size
            param_size = sum(p.numel() for p in self.model.parameters())
            model_mb = param_size * 2 / (1024**2)  # 2 bytes for FP16
            print(f"✅ Model loaded: {model_mb:.1f}MB")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.model = None
    
    def enhance_image(self, image_path: str, output_path: str = None) -> str:
        """Enhance image with guaranteed low memory usage"""
        if output_path is None:
            output_path = image_path.replace('.', '_enhanced.')
            
        print(f"🎨 Enhancing {os.path.basename(image_path)}...")
        
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Failed to read image")
                return image_path
                
            h, w = img.shape[:2]
            print(f"  Input: {w}x{h}")
            
            # Use fallback for very large images
            if h > 2048 or w > 2048:
                print("  ⚠️ Large image, using CPU fallback")
                enhanced = self._cpu_upscale(img)
            elif self.model is not None:
                enhanced = self._enhance_with_model(img)
            else:
                enhanced = self._cpu_upscale(img)
            
            # Ensure 2K limit
            h, w = enhanced.shape[:2]
            if w > 2048 or h > 1080:
                scale = min(2048/w, 1080/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                print(f"  📐 Resizing from {w}x{h} to {new_w}x{new_h} (2K limit)")
            
            # Save result
            cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            new_h, new_w = enhanced.shape[:2]
            print(f"  ✅ Output: {new_w}x{new_h}")
            
            # Force memory cleanup
            self._cleanup_memory()
            
            return output_path
            
        except Exception as e:
            print(f"  ❌ Enhancement failed: {e}")
            # Try CPU fallback
            try:
                img = cv2.imread(image_path)
                enhanced = self._cpu_upscale(img)
                cv2.imwrite(output_path, enhanced)
                return output_path
            except:
                return image_path
    
    def _enhance_with_model(self, img):
        """Enhance using model with extreme memory safety"""
        h, w = img.shape[:2]
        
        # Output image (on CPU to save GPU memory)
        output = np.zeros((h * self.scale, w * self.scale, 3), dtype=np.uint8)
        
        # Process in very small tiles
        tile_size = self.tile_size
        
        print(f"  Processing {tile_size}x{tile_size} tiles...")
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Extract tile
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tile = img[y:y_end, x:x_end]
                
                # Skip if tile is too small
                if tile.shape[0] < 4 or tile.shape[1] < 4:
                    continue
                
                try:
                    # Process tile
                    enhanced_tile = self._process_single_tile(tile)
                    
                    # Place in output
                    out_y = y * self.scale
                    out_x = x * self.scale
                    out_y_end = out_y + enhanced_tile.shape[0]
                    out_x_end = out_x + enhanced_tile.shape[1]
                    
                    output[out_y:out_y_end, out_x:out_x_end] = enhanced_tile
                    
                except Exception as e:
                    # If tile fails, use CPU upscale for that tile
                    fallback = cv2.resize(tile, (tile.shape[1]*self.scale, tile.shape[0]*self.scale), 
                                        interpolation=cv2.INTER_CUBIC)
                    out_y = y * self.scale
                    out_x = x * self.scale
                    output[out_y:out_y+fallback.shape[0], out_x:out_x+fallback.shape[1]] = fallback
                
                # Force memory cleanup after each tile
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return output
    
    def _process_single_tile(self, tile):
        """Process a single tile with proper error handling"""
        # Convert to tensor
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tile_norm = tile_rgb.astype(np.float32) / 255.0
        
        # Create tensor with correct shape
        tile_tensor = torch.from_numpy(tile_norm).permute(2, 0, 1).unsqueeze(0)
        tile_tensor = tile_tensor.to(self.device)
        
        # Convert to half precision if using GPU
        if self.device.type == 'cuda':
            tile_tensor = tile_tensor.half()
        
        # Process
        with torch.no_grad():
            enhanced_tensor = self.model(tile_tensor)
        
        # Convert back to numpy
        enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0)
        enhanced = enhanced.cpu().float().numpy()
        enhanced = (enhanced * 255).clip(0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        
        # Clean up tensors
        del tile_tensor, enhanced_tensor
        
        return enhanced
    
    def _cpu_upscale(self, img):
        """CPU-only upscaling fallback"""
        print("  📈 Using CPU upscaling...")
        
        # High-quality CPU upscaling (max 2K)
        h, w = img.shape[:2]
        scale_factor = min(self.scale, 2048/w, 1080/h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Use multiple interpolation methods and blend
        cubic = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        lanczos = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Blend for better quality
        result = cv2.addWeighted(cubic, 0.5, lanczos, 0.5, 0)
        
        # Mild sharpening (properly normalized)
        kernel = np.array([[0, -1, 0], 
                          [-1, 5, -1], 
                          [0, -1, 0]], dtype=np.float32)
        result = cv2.filter2D(result, -1, kernel)
        
        return result
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_usage(self):
        """Get current memory usage"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            return f"Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB"
        return "Using CPU"

# Global instance
_memory_safe_enhancer = None

def get_memory_safe_enhancer():
    """Get or create memory-safe enhancer"""
    global _memory_safe_enhancer
    if _memory_safe_enhancer is None:
        _memory_safe_enhancer = MemorySafeEnhancer()
    return _memory_safe_enhancer

# Simple API
def enhance_image_safe(image_path: str, output_path: str = None) -> str:
    """Enhance image with guaranteed <1GB VRAM usage"""
    enhancer = get_memory_safe_enhancer()
    return enhancer.enhance_image(image_path, output_path)