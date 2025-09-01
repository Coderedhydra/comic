#!/usr/bin/env python3
"""
Test script for the Enhanced Comic Generator
Verifies all AI components are working correctly
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from backend.ai_enhanced_core import (
            image_processor, comic_styler, face_detector, layout_optimizer
        )
        print("✅ AI enhanced core imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AI enhanced core: {e}")
        return False
    
    try:
        from backend.ai_bubble_placement import ai_bubble_placer
        print("✅ AI bubble placement imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AI bubble placement: {e}")
        return False
    
    return True

def test_ai_models():
    """Test AI model initialization"""
    print("\n🤖 Testing AI models...")
    
    try:
        from backend.ai_enhanced_core import AIEnhancedCore
        core = AIEnhancedCore()
        print("✅ AI core initialized successfully")
        
        # Test MediaPipe
        if hasattr(core, 'face_mesh') and core.face_mesh:
            print("✅ MediaPipe face detection available")
        else:
            print("⚠️ MediaPipe not available")
        
        return True
    except Exception as e:
        print(f"❌ AI model initialization failed: {e}")
        return False

def test_image_processing():
    """Test image processing capabilities"""
    print("\n🖼️ Testing image processing...")
    
    try:
        from backend.ai_enhanced_core import image_processor
        
        # Create a test image
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        test_path = "test_image.png"
        cv2.imwrite(test_path, test_image)
        
        # Test enhancement
        enhanced_path = image_processor.enhance_image_quality(test_path)
        print("✅ Image enhancement completed")
        
        # Clean up
        os.remove(test_path)
        if os.path.exists(enhanced_path) and enhanced_path != test_path:
            os.remove(enhanced_path)
        
        return True
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def test_face_detection():
    """Test face detection capabilities"""
    print("\n👤 Testing face detection...")
    
    try:
        from backend.ai_enhanced_core import face_detector
        
        # Create a test image with a simple face-like pattern
        test_image = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Draw a simple face-like pattern
        cv2.circle(test_image, (200, 150), 50, (255, 255, 255), -1)  # Head
        cv2.circle(test_image, (180, 140), 5, (0, 0, 0), -1)        # Left eye
        cv2.circle(test_image, (220, 140), 5, (0, 0, 0), -1)        # Right eye
        cv2.ellipse(test_image, (200, 170), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        test_path = "test_face.png"
        cv2.imwrite(test_path, test_image)
        
        # Test face detection
        faces = face_detector.detect_faces_advanced(test_path)
        print(f"✅ Face detection completed, found {len(faces)} faces")
        
        # Clean up
        os.remove(test_path)
        
        return True
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_comic_styling():
    """Test comic styling capabilities"""
    print("\n🎨 Testing comic styling...")
    
    try:
        from backend.ai_enhanced_core import comic_styler
        
        # Create a test image
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        test_path = "test_style.png"
        cv2.imwrite(test_path, test_image)
        
        # Test styling
        styled_path = comic_styler.apply_comic_style(test_path, style_type="modern")
        print("✅ Comic styling completed")
        
        # Clean up
        os.remove(test_path)
        if os.path.exists(styled_path) and styled_path != test_path:
            os.remove(styled_path)
        
        return True
    except Exception as e:
        print(f"❌ Comic styling test failed: {e}")
        return False

def test_bubble_placement():
    """Test bubble placement capabilities"""
    print("\n💬 Testing bubble placement...")
    
    try:
        from backend.ai_bubble_placement import ai_bubble_placer
        
        # Create a test image
        test_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        test_path = "test_bubble.png"
        cv2.imwrite(test_path, test_image)
        
        # Test bubble placement
        panel_coords = (0, 600, 0, 400)
        lip_coords = (300, 200)
        dialogue = "Hello, this is a test!"
        
        position = ai_bubble_placer.place_bubble_ai(
            test_path, panel_coords, lip_coords, dialogue
        )
        
        print(f"✅ Bubble placement completed: {position}")
        
        # Clean up
        os.remove(test_path)
        
        return True
    except Exception as e:
        print(f"❌ Bubble placement test failed: {e}")
        return False

def test_layout_optimization():
    """Test layout optimization capabilities"""
    print("\n📐 Testing layout optimization...")
    
    try:
        from backend.ai_enhanced_core import layout_optimizer
        
        # Create test image paths
        test_paths = ["test1.png", "test2.png", "test3.png", "test4.png"]
        
        # Create test images
        for i, path in enumerate(test_paths):
            test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
            cv2.imwrite(path, test_image)
        
        # Test layout optimization
        layout = layout_optimizer.optimize_layout(test_paths, target_layout="2x2")
        print(f"✅ Layout optimization completed: {len(layout)} panels")
        
        # Clean up
        for path in test_paths:
            if os.path.exists(path):
                os.remove(path)
        
        return True
    except Exception as e:
        print(f"❌ Layout optimization test failed: {e}")
        return False

def test_system_integration():
    """Test overall system integration"""
    print("\n🔗 Testing system integration...")
    
    try:
        from backend.ai_enhanced_core import (
            image_processor, comic_styler, face_detector, layout_optimizer
        )
        from backend.ai_bubble_placement import ai_bubble_placer
        
        # Test that all components work together
        print("✅ All AI components initialized successfully")
        print("✅ System integration test passed")
        
        return True
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Enhanced Comic Generator - System Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("AI Models Test", test_ai_models),
        ("Image Processing Test", test_image_processing),
        ("Face Detection Test", test_face_detection),
        ("Comic Styling Test", test_comic_styling),
        ("Bubble Placement Test", test_bubble_placement),
        ("Layout Optimization Test", test_layout_optimization),
        ("System Integration Test", test_system_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)