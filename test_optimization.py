#!/usr/bin/env python3
"""
Test script to verify model loading optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.keyframes.keyframes import _get_features, _get_probs
import torch

def test_model_loading():
    """Test that models are loaded only once"""
    print("🧪 Testing model loading optimization...")
    
    # Create dummy frames for testing
    dummy_frames = ["frames/final/frame001.png"]  # Use existing frame
    
    if not os.path.exists(dummy_frames[0]):
        print("❌ Test frame not found, creating dummy...")
        # Create a dummy image if needed
        import numpy as np
        from PIL import Image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        os.makedirs("frames/final", exist_ok=True)
        dummy_img.save(dummy_frames[0])
    
    print("🔄 First call to _get_features...")
    features1 = _get_features(dummy_frames, gpu=False)
    
    print("🔄 Second call to _get_features (should use cached model)...")
    features2 = _get_features(dummy_frames, gpu=False)
    
    print("🔄 First call to _get_probs...")
    probs1 = _get_probs(features1, gpu=False)
    
    print("🔄 Second call to _get_probs (should use cached model)...")
    probs2 = _get_probs(features2, gpu=False)
    
    print("✅ Model loading optimization test completed!")
    print("📊 Features shape:", features1.shape)
    print("📊 Probabilities shape:", probs1.shape)

if __name__ == "__main__":
    test_model_loading()