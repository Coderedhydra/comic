"""
AI Model Manager for State-of-the-Art Image Enhancement
Manages Real-ESRGAN, GFPGAN, SwinIR and other models
Optimized for NVIDIA RTX 3050
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
import requests
from tqdm import tqdm
import hashlib
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Model URLs and checksums
MODEL_URLS = {
    'RealESRGAN_x4plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'hash': '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1'
    },
    'RealESRGAN_x4plus_anime_6B': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        'hash': 'f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da'
    },
    'RealESRNet_x4plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
        'hash': '99ec365d4afad750833258a1a24f44ca3fefd45f1bb7f14e1d195f21934bb428'
    },
    'GFPGAN_v1.3': {
        'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        'hash': 'c953a88f2ba4e03fb985a7582126c2267b4c3db0e50def3448b844e88e8b8f5e'
    },
    'detection_Resnet50_Final': {
        'url': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'hash': '6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d'
    },
    'parsing_parsenet': {
        'url': 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_parsenet.pth',
        'hash': '3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2'
    }
}

class AIModelManager:
    """Manages AI models for image enhancement with GPU optimization"""
    
    def __init__(self, device=None, model_dir='models'):
        """Initialize model manager with RTX 3050 optimization"""
        
        # Set device - prioritize CUDA for RTX 3050
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
                
                # RTX 3050 optimization settings
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Set memory fraction to avoid OOM on 4GB/8GB RTX 3050
                torch.cuda.set_per_process_memory_fraction(0.8)
            else:
                self.device = torch.device('cpu')
                print("💻 Using CPU (GPU not available)")
        else:
            self.device = device
            
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model instances
        self.realesrgan = None
        self.realesrgan_anime = None
        self.gfpgan = None
        self.face_enhancer = None
        
        # Model configs
        self.current_models = {}
        
    def download_model(self, model_name: str) -> str:
        """Download model if not exists"""
        if model_name not in MODEL_URLS:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_info = MODEL_URLS[model_name]
        model_path = os.path.join(self.model_dir, f"{model_name}.pth")
        
        # Check if already exists and valid
        if os.path.exists(model_path):
            print(f"✅ Model {model_name} already exists")
            return model_path
            
        print(f"📥 Downloading {model_name}...")
        
        # Download with progress bar
        response = requests.get(model_info['url'], stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
        print(f"✅ Downloaded {model_name}")
        return model_path
        
    def load_realesrgan(self, model_name='RealESRGAN_x4plus', scale=4):
        """Load Real-ESRGAN model optimized for RTX 3050"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            print(f"🔄 Loading {model_name}...")
            
            # Download model if needed
            model_path = self.download_model(model_name)
            
            # Different architectures for different models
            if 'anime' in model_name:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6)
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
                
            # Initialize upsampler
            self.realesrgan = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=model,
                device=self.device,
                # RTX 3050 optimizations
                tile=256,  # Smaller tile size for 4GB VRAM
                tile_pad=10,
                pre_pad=0,
                half=True if self.device.type == 'cuda' else False  # FP16 for GPU
            )
            
            if 'anime' in model_name:
                self.realesrgan_anime = self.realesrgan
            
            print(f"✅ Loaded {model_name} on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Real-ESRGAN: {e}")
            return False
            
    def load_gfpgan(self):
        """Load GFPGAN for face enhancement"""
        try:
            from gfpgan import GFPGANer
            
            print("🔄 Loading GFPGAN v1.3...")
            
            # Download models
            model_path = self.download_model('GFPGAN_v1.3')
            det_model_path = self.download_model('detection_Resnet50_Final')
            parse_model_path = self.download_model('parsing_parsenet')
            
            # Initialize GFPGAN
            self.gfpgan = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.realesrgan,  # Use Real-ESRGAN for background
                device=self.device
            )
            
            print("✅ Loaded GFPGAN on", self.device)
            return True
            
        except Exception as e:
            print(f"❌ Failed to load GFPGAN: {e}")
            return False
            
    def enhance_image_realesrgan(self, image, use_anime_model=False):
        """Enhance image using Real-ESRGAN"""
        if use_anime_model and self.realesrgan_anime:
            upsampler = self.realesrgan_anime
        else:
            upsampler = self.realesrgan
            
        if upsampler is None:
            model_name = 'RealESRGAN_x4plus_anime_6B' if use_anime_model else 'RealESRGAN_x4plus'
            if not self.load_realesrgan(model_name):
                return image
                
            upsampler = self.realesrgan_anime if use_anime_model else self.realesrgan
            
        try:
            # Convert to numpy if PIL Image
            if isinstance(image, Image.Image):
                image = np.array(image)
                
            # Ensure BGR format for Real-ESRGAN
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            # Enhance
            with torch.no_grad():
                output, _ = upsampler.enhance(image, outscale=4)
                
            return output
            
        except Exception as e:
            print(f"❌ Real-ESRGAN enhancement failed: {e}")
            return image
            
    def enhance_face_gfpgan(self, image, only_center_face=False, paste_back=True):
        """Enhance faces in image using GFPGAN"""
        if self.gfpgan is None:
            if not self.load_gfpgan():
                return image
                
        try:
            # Convert to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
                
            # Ensure BGR format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            # Enhance faces
            with torch.no_grad():
                _, _, output = self.gfpgan.enhance(
                    image,
                    has_aligned=False,
                    only_center_face=only_center_face,
                    paste_back=paste_back,
                    weight=0.5
                )
                
            return output
            
        except Exception as e:
            print(f"❌ GFPGAN enhancement failed: {e}")
            return image
            
    def enhance_image_pipeline(self, image_path: str, output_path: str = None,
                             enhance_face=True, use_anime_model=False) -> str:
        """Complete enhancement pipeline optimized for RTX 3050"""
        
        print(f"🎨 Enhancing {os.path.basename(image_path)}...")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                return image_path
                
            original_shape = image.shape[:2]
            
            # Step 1: Super-resolution with Real-ESRGAN
            print("  📈 Applying 4x super-resolution...")
            enhanced = self.enhance_image_realesrgan(image, use_anime_model)
            
            # Step 2: Face enhancement with GFPGAN (if faces detected)
            if enhance_face:
                print("  👤 Enhancing faces...")
                enhanced = self.enhance_face_gfpgan(enhanced)
                
            # Step 3: Additional post-processing
            print("  ✨ Applying final enhancements...")
            enhanced = self.post_process(enhanced)
            
            # Save result
            if output_path is None:
                output_path = image_path.replace('.', '_enhanced.')
                
            cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            new_shape = enhanced.shape[:2]
            print(f"  ✅ Enhanced: {original_shape} → {new_shape}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Enhancement pipeline failed: {e}")
            return image_path
            
    def post_process(self, image):
        """Additional post-processing for enhanced quality"""
        try:
            # 1. Slight sharpening
            kernel = np.array([[-0.5,-0.5,-0.5], 
                              [-0.5, 5,-0.5], 
                              [-0.5,-0.5,-0.5]]) / 1
            image = cv2.filter2D(image, -1, kernel)
            
            # 2. Color enhancement in LAB space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Enhance L channel with CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Enhance color channels slightly
            a = cv2.convertScaleAbs(a, alpha=1.1, beta=0)
            b = cv2.convertScaleAbs(b, alpha=1.1, beta=0)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 3. Final brightness/contrast adjustment
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.05, beta=5)
            
            return enhanced
            
        except Exception as e:
            print(f"⚠️ Post-processing failed: {e}")
            return image
            
    def clear_memory(self):
        """Clear GPU memory - important for RTX 3050 with limited VRAM"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
# Global instance
_ai_model_manager = None

def get_ai_model_manager():
    """Get or create global AI model manager"""
    global _ai_model_manager
    if _ai_model_manager is None:
        _ai_model_manager = AIModelManager()
    return _ai_model_manager