#!/usr/bin/env python3
"""
Enhanced Comic Generator - Web Interface Runner
Run this script to start the Flask web interface for comic generation
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['flask', 'yt_dlp', 'opencv-python', 'pillow', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages + ['--break-system-packages'], check=True)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    return True

def check_directories():
    """Ensure required directories exist"""
    directories = ['video', 'frames/final', 'output', 'static', 'templates']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created/verified")

def start_flask_app():
    """Start the Flask web application"""
    print("🚀 Starting Enhanced Comic Generator Web Interface...")
    print("✨ Features:")
    print("   - AI-enhanced image processing")
    print("   - Advanced face detection")
    print("   - Smart bubble placement")
    print("   - High-quality comic styling")
    print("   - Optimized 2x2 layout")
    print("")
    
    # Check if app_enhanced.py exists
    if not os.path.exists('app_enhanced.py'):
        print("❌ app_enhanced.py not found!")
        return False
    
    try:
        # Start Flask app
        print("🌐 Starting Flask server...")
        process = subprocess.Popen([sys.executable, 'app_enhanced.py'])
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            import requests
            response = requests.get('http://localhost:5000', timeout=5)
            if response.status_code == 200:
                print("✅ Flask server started successfully!")
                print("🌐 Web interface available at: http://localhost:5000")
                print("")
                print("📋 How to use:")
                print("   1. Open your browser and go to: http://localhost:5000")
                print("   2. Click 'Upload Video' to select an MP4 file")
                print("   3. Or click 'Enter Link' to paste a YouTube URL")
                print("   4. Click 'Submit' to generate your comic")
                print("   5. The comic will automatically open in your browser")
                print("")
                print("🔄 The server will continue running in the background")
                print("🛑 Press Ctrl+C to stop the server")
                
                # Try to open browser automatically
                try:
                    webbrowser.open('http://localhost:5000')
                    print("🌐 Browser opened automatically!")
                except:
                    print("📱 Please open http://localhost:5000 manually in your browser")
                
                return process
            else:
                print(f"❌ Server returned status code: {response.status_code}")
                return False
        except ImportError:
            print("⚠️ requests module not available, skipping server check")
            print("🌐 Web interface should be available at: http://localhost:5000")
            return process
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")
        return False

def main():
    """Main function"""
    print("🎬 Enhanced Comic Generator - Web Interface")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Cannot proceed without dependencies")
        return
    
    # Check directories
    check_directories()
    
    # Start Flask app
    process = start_flask_app()
    
    if process:
        try:
            # Keep the script running
            print("\n🔄 Server is running... Press Ctrl+C to stop")
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            process.terminate()
            process.wait()
            print("✅ Server stopped")
    else:
        print("❌ Failed to start web interface")

if __name__ == '__main__':
    main()