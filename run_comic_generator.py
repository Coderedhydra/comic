#!/usr/bin/env python3
"""
Simple Comic Generator Runner
Runs the enhanced comic generator with better error handling
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'opencv-python', 'pillow', 'numpy', 
        'mediapipe', 'torch', 'transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_enhanced.txt")
        return False
    
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("💻 No GPU detected, using CPU")
            return False
    except:
        print("💻 GPU check failed, using CPU")
        return False

def run_flask_app():
    """Run the Flask app"""
    print("\n🚀 Starting Comic Generator...")
    print("📱 Web interface: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        # Set environment variables for better performance
        os.environ['AI_ENHANCED'] = '1'
        os.environ['HIGH_QUALITY'] = '1'
        
        # Run the Flask app
        from app_enhanced import app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopping Comic Generator...")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🎨 Enhanced Comic Generator")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return False
    
    # Check GPU
    check_gpu()
    
    # Run the app
    return run_flask_app()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)