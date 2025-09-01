#!/bin/bash
# Installation script for AI models
# Optimized for NVIDIA RTX 3050

echo "🚀 AI Model Installation Script"
echo "==============================="
echo "This script will install Real-ESRGAN, GFPGAN and other AI models"
echo "Optimized for NVIDIA RTX 3050 GPU"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check CUDA availability
echo ""
echo "🎮 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "❌ NVIDIA GPU not detected"

# Create virtual environment (recommended)
echo ""
echo "📦 Setting up virtual environment..."
if [ ! -d "venv_ai" ]; then
    python3 -m venv venv_ai
    echo "✅ Created virtual environment: venv_ai"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv_ai/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support for RTX 3050
echo ""
echo "🔥 Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install core requirements
echo ""
echo "📦 Installing core requirements..."
pip install -r requirements_enhanced.txt

# Install Real-ESRGAN
echo ""
echo "🎨 Installing Real-ESRGAN..."
pip install basicsr>=1.4.2
pip install realesrgan>=0.3.0

# Install GFPGAN
echo ""
echo "👤 Installing GFPGAN..."
pip install facexlib>=0.3.0
pip install gfpgan>=1.3.8

# Install additional AI libraries
echo ""
echo "🤖 Installing additional AI libraries..."
pip install opencv-python>=4.8.0
pip install opencv-contrib-python>=4.8.0
pip install albumentations>=1.3.1
pip install psutil gpustat py3nvml

# Create models directory
echo ""
echo "📁 Creating models directory..."
mkdir -p models

# Download models (optional - will download on first use)
echo ""
echo "📥 Pre-downloading models (optional)..."
echo "Models will be automatically downloaded on first use"
echo "To pre-download, run: python3 -c 'from backend.ai_model_manager import AIModelManager; m = AIModelManager(); m.download_model(\"RealESRGAN_x4plus\")'"

# Test installation
echo ""
echo "🧪 Testing installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import realesrgan; print('Real-ESRGAN: Installed ✅')"
python3 -c "import gfpgan; print('GFPGAN: Installed ✅')"

echo ""
echo "✅ Installation complete!"
echo ""
echo "To use the AI models:"
echo "1. Activate the virtual environment: source venv_ai/bin/activate"
echo "2. Run the test script: python test_ai_models.py"
echo "3. Use in your application with USE_AI_MODELS=1 environment variable"
echo ""
echo "For RTX 3050 optimization tips:"
echo "- The models are configured to use FP16 (half precision) for better performance"
echo "- Tile size is set to 256 to fit in 4GB/8GB VRAM"
echo "- Memory fraction is limited to 80% to prevent OOM errors"
echo ""