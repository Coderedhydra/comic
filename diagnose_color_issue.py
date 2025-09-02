#!/usr/bin/env python3
"""
Diagnose the green/colorless image issue
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_image_colors(image_path):
    """Analyze color distribution in an image"""
    
    print(f"\n🔍 Analyzing: {image_path}")
    print("=" * 50)
    
    if not os.path.exists(image_path):
        print("❌ Image not found")
        return
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Failed to read image")
        return
    
    h, w = img.shape[:2]
    print(f"📐 Dimensions: {w}x{h}")
    
    # Analyze color channels
    b, g, r = cv2.split(img)
    
    print(f"\n📊 Channel Statistics:")
    print(f"  Blue:  mean={b.mean():.1f}, std={b.std():.1f}, min={b.min()}, max={b.max()}")
    print(f"  Green: mean={g.mean():.1f}, std={g.std():.1f}, min={g.min()}, max={g.max()}")
    print(f"  Red:   mean={r.mean():.1f}, std={r.std():.1f}, min={r.min()}, max={r.max()}")
    
    # Check for channel imbalance
    channel_means = [b.mean(), g.mean(), r.mean()]
    max_diff = max(channel_means) - min(channel_means)
    
    if max_diff > 30:
        print(f"\n⚠️ Channel imbalance detected! Difference: {max_diff:.1f}")
        dominant = ['Blue', 'Green', 'Red'][channel_means.index(max(channel_means))]
        print(f"  Dominant channel: {dominant}")
    else:
        print(f"\n✅ Color balance looks normal (diff: {max_diff:.1f})")
    
    # Check if image is mostly gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_diff = np.mean(np.abs(img - gray[:,:,np.newaxis]))
    
    if color_diff < 10:
        print(f"\n⚠️ Image appears to be grayscale (color diff: {color_diff:.1f})")
    else:
        print(f"\n✅ Image has color information (color diff: {color_diff:.1f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Color channels
    axes[0, 1].imshow(r, cmap='Reds')
    axes[0, 1].set_title('Red Channel')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(g, cmap='Greens')
    axes[1, 0].set_title('Green Channel')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(b, cmap='Blues')
    axes[1, 1].set_title('Blue Channel')
    axes[1, 1].axis('off')
    
    # Save analysis
    output_path = image_path.replace('.png', '_color_analysis.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"\n📊 Saved color analysis: {output_path}")

def compare_before_after():
    """Compare frames before and after enhancement"""
    
    print("\n🔬 Comparing Enhancement Effects")
    print("=" * 50)
    
    # Check for test frames
    test_frames = []
    if os.path.exists('frames/final'):
        frames = [f for f in os.listdir('frames/final') if f.endswith('.png')]
        if frames:
            test_frames = [os.path.join('frames/final', frames[0])]
            if len(frames) > 20:
                test_frames.append(os.path.join('frames/final', frames[20]))
    
    for frame_path in test_frames:
        analyze_image_colors(frame_path)

if __name__ == "__main__":
    print("🎨 Color Issue Diagnostic Tool")
    print("=" * 50)
    
    # Run diagnostics
    compare_before_after()
    
    print("\n💡 Recommendations:")
    print("1. If green tint: Check color channel processing")
    print("2. If grayscale: Check color space conversions")
    print("3. If channel imbalance: Check enhancement algorithms")
    print("\n✅ The fixed enhancement should preserve original colors!")