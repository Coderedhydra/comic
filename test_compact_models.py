#!/usr/bin/env python3
"""
Test Compact AI Models (SwinIR & Real-ESRGAN)
Designed for RTX 3050 Laptop GPU with <1GB VRAM usage
"""

import os
import sys
import time
import cv2
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.compact_ai_models import CompactAIEnhancer, create_compact_enhancer
from backend.advanced_image_enhancer import AdvancedImageEnhancer

def print_gpu_info():
    """Print GPU information"""
    print("=" * 60)
    print("🎮 GPU INFORMATION")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024**3)
        print(f"Total VRAM: {total_vram:.1f} GB")
        
        # Current usage
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"Currently allocated: {allocated:.1f} MB")
        print(f"Currently reserved: {reserved:.1f} MB")
    else:
        print("No GPU detected, using CPU")
    
    print("=" * 60)

def create_test_image():
    """Create a test image"""
    print("\n🎨 Creating test image...")
    
    # Create a 256x256 test image (small to test quickly)
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    
    # Add some features
    # Circle (face-like)
    cv2.circle(img, (128, 100), 40, (200, 150, 100), -1)
    
    # Eyes
    cv2.circle(img, (115, 90), 8, (50, 50, 50), -1)
    cv2.circle(img, (141, 90), 8, (50, 50, 50), -1)
    
    # Smile
    cv2.ellipse(img, (128, 115), (20, 10), 0, 0, 180, (50, 50, 50), 2)
    
    # Add text
    cv2.putText(img, "AI Test", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add some noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Save
    test_path = "test_compact.jpg"
    cv2.imwrite(test_path, img)
    print(f"✅ Test image created: {test_path}")
    
    return test_path

def test_swinir():
    """Test SwinIR lightweight model"""
    print("\n🧪 TESTING SWINIR LIGHTWEIGHT")
    print("=" * 60)
    
    # Create enhancer
    enhancer = CompactAIEnhancer(model_type='swinir')
    
    # Create test image
    test_image = create_test_image()
    
    # Show memory before
    print(f"\nMemory before enhancement: {enhancer.get_memory_usage()}")
    
    # Enhance
    start = time.time()
    result = enhancer.enhance_image(test_image, "test_swinir_result.jpg")
    elapsed = time.time() - start
    
    # Show results
    print(f"\nProcessing time: {elapsed:.2f}s")
    print(f"Memory after enhancement: {enhancer.get_memory_usage()}")
    
    # Check output
    if os.path.exists(result):
        img = cv2.imread(result)
        print(f"Output size: {img.shape[1]}x{img.shape[0]}")
        print("✅ SwinIR test passed!")
    else:
        print("❌ SwinIR test failed!")
        
    return result

def test_compact_realesrgan():
    """Test compact Real-ESRGAN model"""
    print("\n🧪 TESTING COMPACT REAL-ESRGAN")
    print("=" * 60)
    
    # Create enhancer
    enhancer = CompactAIEnhancer(model_type='realesrgan')
    
    # Create test image
    test_image = create_test_image()
    
    # Show memory before
    print(f"\nMemory before enhancement: {enhancer.get_memory_usage()}")
    
    # Enhance
    start = time.time()
    result = enhancer.enhance_image(test_image, "test_realesrgan_result.jpg")
    elapsed = time.time() - start
    
    # Show results
    print(f"\nProcessing time: {elapsed:.2f}s")
    print(f"Memory after enhancement: {enhancer.get_memory_usage()}")
    
    # Check output
    if os.path.exists(result):
        img = cv2.imread(result)
        print(f"Output size: {img.shape[1]}x{img.shape[0]}")
        print("✅ Real-ESRGAN test passed!")
    else:
        print("❌ Real-ESRGAN test failed!")
        
    return result

def test_advanced_enhancer():
    """Test the integrated advanced enhancer"""
    print("\n🧪 TESTING INTEGRATED ENHANCER")
    print("=" * 60)
    
    # This will automatically use compact models for <6GB VRAM
    enhancer = AdvancedImageEnhancer()
    
    # Create test image
    test_image = create_test_image()
    
    # Enhance
    start = time.time()
    result = enhancer.enhance_image(test_image, "test_integrated_result.jpg")
    elapsed = time.time() - start
    
    print(f"\nProcessing time: {elapsed:.2f}s")
    print("✅ Integrated enhancer test complete!")
    
    return result

def test_memory_efficiency():
    """Test memory usage stays under 1GB"""
    print("\n💾 TESTING MEMORY EFFICIENCY")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("No GPU available for memory test")
        return
        
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    initial = torch.cuda.memory_allocated() / (1024**2)
    print(f"Initial memory: {initial:.1f} MB")
    
    # Test SwinIR
    enhancer1 = CompactAIEnhancer(model_type='swinir')
    after_load = torch.cuda.memory_allocated() / (1024**2)
    print(f"After loading SwinIR: {after_load:.1f} MB")
    
    # Process image
    test_image = create_test_image()
    enhancer1.enhance_image(test_image, "temp.jpg")
    after_process = torch.cuda.memory_allocated() / (1024**2)
    print(f"After processing: {after_process:.1f} MB")
    
    # Clean up
    del enhancer1
    torch.cuda.empty_cache()
    final = torch.cuda.memory_allocated() / (1024**2)
    print(f"After cleanup: {final:.1f} MB")
    
    print(f"\n✅ Peak memory usage: {after_process:.1f} MB (Target: <1000 MB)")
    if after_process < 1000:
        print("✅ Memory efficiency test PASSED!")
    else:
        print("⚠️ Memory usage higher than expected")

def run_all_tests():
    """Run all compact model tests"""
    print("\n🚀 COMPACT AI MODEL TEST SUITE")
    print("For RTX 3050 Laptop GPU (<1GB VRAM)")
    print("=" * 60)
    
    try:
        # Print GPU info
        print_gpu_info()
        
        # Test SwinIR
        swinir_result = test_swinir()
        
        # Clear memory between tests
        torch.cuda.empty_cache()
        
        # Test Real-ESRGAN
        realesrgan_result = test_compact_realesrgan()
        
        # Clear memory
        torch.cuda.empty_cache()
        
        # Test integrated enhancer
        integrated_result = test_advanced_enhancer()
        
        # Test memory efficiency
        test_memory_efficiency()
        
        print("\n✅ ALL TESTS COMPLETED!")
        
        # Cleanup
        print("\n🧹 Cleaning up test files...")
        test_files = [
            "test_compact.jpg",
            "test_swinir_result.jpg", 
            "test_realesrgan_result.jpg",
            "test_integrated_result.jpg",
            "temp.jpg"
        ]
        
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"  Removed: {file}")
                
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()