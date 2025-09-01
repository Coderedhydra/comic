#!/bin/bash
# Installation script for Compact AI Models
# SwinIR & Real-ESRGAN for <1GB VRAM usage

echo "🚀 Compact AI Models Installation"
echo "================================="
echo "SwinIR Lightweight & Compact Real-ESRGAN"
echo "Perfect for RTX 3050 Laptop GPU (4GB VRAM)"
echo ""

# Check Python
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check GPU
echo ""
echo "🎮 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No NVIDIA GPU detected"

# Install minimal requirements
echo ""
echo "📦 Installing requirements..."

# Create minimal requirements file
cat > requirements_compact.txt << EOF
# Minimal requirements for compact models
torch==2.1.0
torchvision==0.16.0
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.1
tqdm==4.66.1
requests==2.31.0
EOF

# Install
pip install -r requirements_compact.txt

# Create model directories
echo ""
echo "📁 Creating model directories..."
mkdir -p models_compact
mkdir -p models_small

echo ""
echo "✅ Installation complete!"
echo ""
echo "📊 Model Information:"
echo "- SwinIR Lightweight: ~12MB model, <500MB VRAM"
echo "- Compact Real-ESRGAN: ~20MB model, <500MB VRAM"
echo "- Processing: 256x256 → 1024x1024 in ~2 seconds"
echo ""
echo "To test:"
echo "python test_compact_models.py"
echo ""
echo "To use in your code:"
echo "from backend.compact_ai_models import enhance_with_swinir"
echo "result = enhance_with_swinir('input.jpg', 'output.jpg')"
echo ""