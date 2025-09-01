#!/bin/bash
# Lightweight AI Enhancement Installation
# Works with <4GB VRAM (RTX 3050 Laptop GPU)

echo "🚀 Lightweight AI Enhancement Installation"
echo "========================================="
echo "Optimized for RTX 3050 Laptop GPU (4GB VRAM)"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check CUDA availability
echo ""
echo "🎮 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "❌ NVIDIA GPU not detected"

# Create virtual environment
echo ""
echo "📦 Setting up virtual environment..."
if [ ! -d "venv_lightweight" ]; then
    python3 -m venv venv_lightweight
    echo "✅ Created virtual environment: venv_lightweight"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv_lightweight/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support (smaller footprint)
echo ""
echo "🔥 Installing PyTorch (CPU+CUDA lightweight)..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install minimal requirements
echo ""
echo "📦 Installing core requirements..."
cat > requirements_lightweight.txt << EOF
# Lightweight requirements for <4GB VRAM
Flask==2.3.3
Pillow==10.0.1
opencv-python==4.8.1.78
numpy==1.24.3
srt==3.5.2
yt-dlp==2023.10.13
tqdm==4.66.1
requests==2.31.0
psutil==5.9.5
gpustat==1.1.0
scipy==1.11.3
scikit-image==0.21.0
EOF

pip install -r requirements_lightweight.txt

# Test installation
echo ""
echo "🧪 Testing installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

echo ""
echo "✅ Lightweight installation complete!"
echo ""
echo "Features:"
echo "- 4x AI-enhanced upscaling (memory efficient)"
echo "- Face enhancement without heavy models"
echo "- Optimized for 4GB VRAM"
echo "- Fast processing with quality output"
echo ""
echo "To use:"
echo "1. Activate environment: source venv_lightweight/bin/activate"
echo "2. Run: python app_enhanced.py"
echo ""