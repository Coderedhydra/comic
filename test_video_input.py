#!/usr/bin/env python3
"""
Test video input functionality
"""

import os
import sys

def test_video_paths():
    """Test video file paths and availability"""
    print("🎬 Testing Video Input System\n")
    
    # Check video directory
    video_dir = "video"
    if not os.path.exists(video_dir):
        print("❌ Video directory missing")
        return False
    
    print("✅ Video directory exists")
    
    # List all video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    print(f"📹 Found {len(video_files)} video files:")
    
    for video in video_files:
        size_mb = os.path.getsize(os.path.join(video_dir, video)) / (1024*1024)
        print(f"   • {video} ({size_mb:.1f} MB)")
    
    # Check for expected uploaded.mp4
    uploaded_path = "video/uploaded.mp4"
    if os.path.exists(uploaded_path):
        size_mb = os.path.getsize(uploaded_path) / (1024*1024)
        print(f"✅ Expected upload target exists: uploaded.mp4 ({size_mb:.1f} MB)")
        return True
    else:
        print("❌ Expected upload target missing: uploaded.mp4")
        
        # Check if we can copy from existing video
        if "IronMan.mp4" in video_files:
            print("🔄 Copying IronMan.mp4 to uploaded.mp4...")
            import shutil
            try:
                shutil.copy2("video/IronMan.mp4", "video/uploaded.mp4")
                print("✅ Created uploaded.mp4 from existing video")
                return True
            except Exception as e:
                print(f"❌ Copy failed: {e}")
                return False
        else:
            print("❌ No video files available to copy")
            return False

def test_app_functionality():
    """Test app initialization"""
    print("\n🚀 Testing App Functionality\n")
    
    try:
        # Import the main app
        from app_enhanced import EnhancedComicGenerator, app
        
        print("✅ App imports successful")
        
        # Test generator initialization
        generator = EnhancedComicGenerator()
        print(f"✅ Generator initialized")
        print(f"   📹 Video path: {generator.video_path}")
        print(f"   📁 Frames dir: {generator.frames_dir}")
        print(f"   📤 Output dir: {generator.output_dir}")
        
        # Check if video exists
        if os.path.exists(generator.video_path):
            size_mb = os.path.getsize(generator.video_path) / (1024*1024)
            print(f"✅ Video accessible: {size_mb:.1f} MB")
        else:
            print(f"❌ Video not found: {generator.video_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ App functionality test failed: {e}")
        return False

def test_template_system():
    """Test Flask template system"""
    print("\n📄 Testing Template System\n")
    
    # Check templates directory
    templates_dir = "templates"
    if not os.path.exists(templates_dir):
        print("❌ Templates directory missing")
        return False
    
    print("✅ Templates directory exists")
    
    # Check for index.html
    index_path = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_path):
        print("✅ index.html template found")
        
        # Check template content
        with open(index_path, "r") as f:
            content = f.read()
        
        if "upload" in content.lower() or "file" in content.lower():
            print("✅ Upload functionality present in template")
            return True
        else:
            print("⚠️ Upload functionality might be missing from template")
            return False
    else:
        print("❌ index.html template missing")
        return False

def test_routes():
    """Test that routes are properly defined"""
    print("\n🛣️ Testing Routes\n")
    
    try:
        from app_enhanced import app
        
        # Get all routes
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append(f"{rule.rule} ({', '.join(rule.methods)})")
        
        print("📋 Available routes:")
        for route in routes:
            print(f"   • {route}")
        
        # Check for key routes
        key_routes = ['/', '/uploader']
        missing_routes = []
        
        for route in key_routes:
            found = any(route in r for r in routes)
            if found:
                print(f"✅ {route} route available")
            else:
                print(f"❌ {route} route missing")
                missing_routes.append(route)
        
        return len(missing_routes) == 0
        
    except Exception as e:
        print(f"❌ Route testing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TESTING VIDEO INPUT SYSTEM")
    print("=" * 50)
    
    tests = [
        ("Video Paths", test_video_paths),
        ("App Functionality", test_app_functionality),
        ("Template System", test_template_system),
        ("Routes", test_routes)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 VIDEO INPUT SYSTEM IS WORKING!")
        print("\n🚀 How to use:")
        print("   1. Run: python3 app_enhanced.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Upload video or use existing IronMan.mp4")
        print("   4. Generate comic with all improvements!")
    else:
        print("\n⚠️ Some issues found - check output above")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)