#!/usr/bin/env python3
"""
Test script for AI model integration
Validates Real-ESRGAN, GFPGAN and other models
Optimized for NVIDIA RTX 3050
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image
import psutil
import GPUtil

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.ai_model_manager import AIModelManager
from backend.advanced_image_enhancer import AdvancedImageEnhancer

def print_system_info():
    """Print system and GPU information"""
    print("=" * 60)
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 60)
    
    # CPU info
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\n🎮 GPU INFORMATION")
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        
        # Current GPU usage
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"GPU Usage: {gpu.load * 100:.1f}%")
            print(f"GPU Memory Used: {gpu.memoryUsed:.1f} MB / {gpu.memoryTotal:.1f} MB")
            print(f"GPU Temperature: {gpu.temperature}°C")
    else:
        print("\n❌ No GPU detected")
    
    print("=" * 60)

def test_model_loading():
    """Test loading of AI models"""
    print("\n🧪 TESTING MODEL LOADING")
    print("=" * 60)
    
    manager = AIModelManager()
    
    # Test Real-ESRGAN loading
    print("\n1. Testing Real-ESRGAN...")
    start = time.time()
    success = manager.load_realesrgan('RealESRGAN_x4plus')
    load_time = time.time() - start
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
    print(f"   Load time: {load_time:.2f}s")
    
    # Test Real-ESRGAN Anime model
    print("\n2. Testing Real-ESRGAN Anime...")
    start = time.time()
    success = manager.load_realesrgan('RealESRGAN_x4plus_anime_6B')
    load_time = time.time() - start
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
    print(f"   Load time: {load_time:.2f}s")
    
    # Test GFPGAN loading
    print("\n3. Testing GFPGAN...")
    start = time.time()
    success = manager.load_gfpgan()
    load_time = time.time() - start
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
    print(f"   Load time: {load_time:.2f}s")
    
    # Clear GPU memory
    manager.clear_memory()
    
    return manager

def create_test_image():
    """Create a test image with faces and details"""
    print("\n🎨 Creating test image...")
    
    # Create a 512x512 test image
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Add some features
    # Face-like circle
    cv2.circle(img, (256, 200), 80, (200, 150, 100), -1)
    # Eyes
    cv2.circle(img, (230, 180), 15, (50, 50, 50), -1)
    cv2.circle(img, (282, 180), 15, (50, 50, 50), -1)
    # Mouth
    cv2.ellipse(img, (256, 230), (40, 20), 0, 0, 180, (50, 50, 50), 2)
    
    # Add some text
    cv2.putText(img, "AI Test", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Save test image
    test_path = "test_image.jpg"
    cv2.imwrite(test_path, img)
    print(f"   Test image saved: {test_path}")
    
    return test_path

def test_enhancement_pipeline(manager, test_image_path):
    """Test the complete enhancement pipeline"""
    print("\n🔬 TESTING ENHANCEMENT PIPELINE")
    print("=" * 60)
    
    # Test 1: Basic enhancement
    print("\n1. Testing basic Real-ESRGAN enhancement...")
    start = time.time()
    
    img = cv2.imread(test_image_path)
    enhanced = manager.enhance_image_realesrgan(img)
    
    process_time = time.time() - start
    
    print(f"   Original size: {img.shape}")
    print(f"   Enhanced size: {enhanced.shape}")
    print(f"   Processing time: {process_time:.2f}s")
    print(f"   Speed: {1/process_time:.2f} FPS")
    
    cv2.imwrite("test_enhanced_realesrgan.jpg", enhanced)
    
    # Test 2: Anime model enhancement
    print("\n2. Testing anime model enhancement...")
    start = time.time()
    
    enhanced_anime = manager.enhance_image_realesrgan(img, use_anime_model=True)
    
    process_time = time.time() - start
    print(f"   Processing time: {process_time:.2f}s")
    
    cv2.imwrite("test_enhanced_anime.jpg", enhanced_anime)
    
    # Test 3: Face enhancement
    print("\n3. Testing GFPGAN face enhancement...")
    start = time.time()
    
    enhanced_face = manager.enhance_face_gfpgan(enhanced)
    
    process_time = time.time() - start
    print(f"   Processing time: {process_time:.2f}s")
    
    cv2.imwrite("test_enhanced_gfpgan.jpg", enhanced_face)
    
    # Test 4: Complete pipeline
    print("\n4. Testing complete enhancement pipeline...")
    start = time.time()
    
    final_enhanced = manager.enhance_image_pipeline(
        test_image_path,
        "test_final_enhanced.jpg",
        enhance_face=True,
        use_anime_model=False
    )
    
    process_time = time.time() - start
    print(f"   Total processing time: {process_time:.2f}s")
    print(f"   Output: {final_enhanced}")
    
    # Clear GPU memory
    manager.clear_memory()

def test_advanced_enhancer():
    """Test the AdvancedImageEnhancer integration"""
    print("\n🎯 TESTING ADVANCED IMAGE ENHANCER")
    print("=" * 60)
    
    # Set environment to use AI models
    os.environ['USE_AI_MODELS'] = '1'
    os.environ['ENHANCE_FACES'] = '1'
    
    enhancer = AdvancedImageEnhancer()
    
    # Create test image if not exists
    if not os.path.exists("test_image.jpg"):
        test_image = create_test_image()
    else:
        test_image = "test_image.jpg"
    
    # Test enhancement
    print("\nTesting integrated enhancement...")
    start = time.time()
    
    result = enhancer.enhance_image(test_image, "test_integrated_enhanced.jpg")
    
    process_time = time.time() - start
    print(f"Processing time: {process_time:.2f}s")
    print(f"Result: {result}")

def test_memory_usage():
    """Test GPU memory usage"""
    print("\n💾 TESTING MEMORY USAGE")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ GPU not available for memory testing")
        return
    
    manager = AIModelManager()
    
    # Initial memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / (1024**2)
    print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    # After loading models
    manager.load_realesrgan('RealESRGAN_x4plus')
    model_memory = torch.cuda.memory_allocated() / (1024**2)
    print(f"After loading Real-ESRGAN: {model_memory:.1f} MB")
    
    # After processing
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    enhanced = manager.enhance_image_realesrgan(img)
    process_memory = torch.cuda.memory_allocated() / (1024**2)
    print(f"After processing: {process_memory:.1f} MB")
    
    # After cleanup
    manager.clear_memory()
    cleanup_memory = torch.cuda.memory_allocated() / (1024**2)
    print(f"After cleanup: {cleanup_memory:.1f} MB")
    
    print(f"\nMemory efficiency: {(1 - cleanup_memory/process_memory)*100:.1f}% recovered")

def run_all_tests():
    """Run all tests"""
    print("\n🚀 AI MODEL INTEGRATION TEST SUITE")
    print("For NVIDIA RTX 3050 Optimization")
    print("=" * 60)
    
    try:
        # System info
        print_system_info()
        
        # Model loading tests
        manager = test_model_loading()
        
        # Create test image
        test_image = create_test_image()
        
        # Enhancement tests
        test_enhancement_pipeline(manager, test_image)
        
        # Advanced enhancer tests
        test_advanced_enhancer()
        
        # Memory tests
        test_memory_usage()
        
        print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        
        # Cleanup
        print("\n🧹 Cleaning up test files...")
        test_files = [
            "test_image.jpg",
            "test_enhanced_realesrgan.jpg",
            "test_enhanced_anime.jpg",
            "test_enhanced_gfpgan.jpg",
            "test_final_enhanced.jpg",
            "test_integrated_enhanced.jpg"
        ]
        
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"   Removed: {file}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()