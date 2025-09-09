#!/usr/bin/env python3
"""
Test Flask upload functionality for Unity Comic Generator
"""

import os
import json
import requests
import time

def test_flask_upload():
    """Test the Flask upload functionality"""
    print("🧪 Testing Flask Upload Functionality")
    print("=" * 50)
    
    # Test if Flask app is running
    try:
        response = requests.get('http://localhost:5000/unity-upload', timeout=5)
        if response.status_code == 200:
            print("✅ Flask app is running")
        else:
            print(f"❌ Flask app returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Flask app not accessible: {e}")
        print("💡 Make sure to run: python3 app_enhanced.py")
        return False
    
    # Test upload page
    try:
        response = requests.get('http://localhost:5000/unity-upload')
        if response.status_code == 200:
            print("✅ Unity upload page accessible")
        else:
            print(f"❌ Upload page returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing upload page: {e}")
        return False
    
    # Test status endpoint (should return 404 for non-existent job)
    try:
        response = requests.get('http://localhost:5000/unity-status/test-job-id')
        if response.status_code == 404:
            print("✅ Status endpoint working (correctly returns 404 for non-existent job)")
        else:
            print(f"⚠️ Status endpoint returned unexpected status {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing status endpoint: {e}")
        return False
    
    print("\n🎉 Flask Upload Functionality Test Complete!")
    print("📝 Manual Testing Steps:")
    print("1. Open http://localhost:5000/unity-upload in browser")
    print("2. Upload a video file")
    print("3. Set number of pages (1-100)")
    print("4. Click 'Generate Unity Comic'")
    print("5. Monitor progress in real-time")
    print("6. Download generated files when complete")
    
    return True

def test_standalone_flask_app():
    """Test the standalone Flask app"""
    print("\n🧪 Testing Standalone Flask App")
    print("=" * 50)
    
    # Test if standalone Flask app is running
    try:
        response = requests.get('http://localhost:5001/', timeout=5)
        if response.status_code == 200:
            print("✅ Standalone Flask app is running")
        else:
            print(f"❌ Standalone Flask app returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Standalone Flask app not accessible: {e}")
        print("💡 Make sure to run: python3 unity_comic_flask.py")
        return False
    
    # Test list jobs endpoint
    try:
        response = requests.get('http://localhost:5001/list-jobs')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ List jobs endpoint working (found {len(data.get('jobs', []))} jobs)")
        else:
            print(f"❌ List jobs endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing list jobs endpoint: {e}")
        return False
    
    print("\n🎉 Standalone Flask App Test Complete!")
    print("📝 Manual Testing Steps:")
    print("1. Open http://localhost:5001/ in browser")
    print("2. Upload a video file with drag & drop or file picker")
    print("3. Configure generation options")
    print("4. Monitor real-time progress")
    print("5. Download individual files or complete ZIP")
    
    return True

if __name__ == "__main__":
    print("🎮 Unity Comic Flask Upload Test")
    print("=" * 50)
    
    # Test integrated Flask app
    integrated_success = test_flask_upload()
    
    # Test standalone Flask app
    standalone_success = test_standalone_flask_app()
    
    print("\n📊 Test Results:")
    print(f"Integrated Flask App: {'✅ PASS' if integrated_success else '❌ FAIL'}")
    print(f"Standalone Flask App: {'✅ PASS' if standalone_success else '❌ FAIL'}")
    
    if integrated_success or standalone_success:
        print("\n🎉 At least one Flask app is working!")
        print("🚀 Ready for Unity comic generation with file uploads!")
    else:
        print("\n❌ Both Flask apps failed!")
        print("💡 Make sure to start one of the Flask apps:")
        print("   - python3 app_enhanced.py (integrated)")
        print("   - python3 unity_comic_flask.py (standalone)")