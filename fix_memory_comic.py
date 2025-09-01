#!/usr/bin/env python3
"""
Quick fix script to run comic generation with memory-safe settings
"""

import os
import sys

# Set environment variables BEFORE importing anything else
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['USE_AI_MODELS'] = '1'
os.environ['ENHANCE_FACES'] = '0'  # Disable face enhancement to save memory

# Force memory fraction
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.4)  # Use only 40% of VRAM

# Now import and run
from app_enhanced import app, EnhancedComicGenerator

def generate_comic_safe(video_path='video/uploaded.mp4'):
    """Generate comic with memory safety"""
    print("🚀 Starting memory-safe comic generation...")
    print(f"📊 GPU Memory limit: 40% of available VRAM")
    
    try:
        generator = EnhancedComicGenerator()
        generator.video_path = video_path
        
        # Override some settings for memory safety
        generator.quality_mode = '0'  # Disable high quality mode
        generator.ai_mode = '1'  # Keep AI mode but with memory limits
        
        # Generate comic
        result = generator.generate_comic()
        
        print("✅ Comic generation complete!")
        return result
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

if __name__ == "__main__":
    # Run the generation
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'video/uploaded.mp4'
        
    generate_comic_safe(video_path)