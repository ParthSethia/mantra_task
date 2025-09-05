#!/usr/bin/env python3
"""
Test script for Flask API endpoints
"""
import requests
import json
import time
import os

API_BASE = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_upload_video(video_path):
    """Test video upload endpoint"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None
    
    try:
        with open(video_path, 'rb') as f:
            files = {'video': f}
            data = {}
            
            print(f"📤 Uploading video: {video_path}")
            response = requests.post(f"{API_BASE}/api/upload-video", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Video upload successful")
            print(f"   Session ID: {result.get('session_id')}")
            print(f"   Message: {result.get('message')[:100]}...")
            return result.get('session_id')
        else:
            print(f"❌ Video upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Video upload error: {e}")
        return None

def test_chat(session_id, message):
    """Test chat endpoint"""
    try:
        data = {
            'session_id': session_id,
            'message': message
        }
        
        print(f"💬 Sending message: {message}")
        response = requests.post(f"{API_BASE}/api/chat", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat response received")
            print(f"   Response: {result.get('response')[:100]}...")
            return True
        else:
            print(f"❌ Chat failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return False

def test_session_info(session_id):
    """Test session info endpoint"""
    try:
        response = requests.get(f"{API_BASE}/api/session/{session_id}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Session info retrieved")
            print(f"   Filename: {result.get('filename')}")
            print(f"   Status: {result.get('status')}")
            return True
        else:
            print(f"❌ Session info failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Session info error: {e}")
        return False

def test_moments(session_id):
    """Test moments endpoint"""
    try:
        response = requests.get(f"{API_BASE}/api/session/{session_id}/moments")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Important moments retrieved")
            return True
        else:
            print(f"❌ Moments failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Moments error: {e}")
        return False

def main():
    """Run API tests"""
    print("=" * 60)
    print("🧪 TESTING FLASK API ENDPOINTS")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("❌ Server not responding. Make sure to run 'python app.py' first")
        return
    
    print()
    
    # Find test video
    test_videos = [
        "data/test_sample5.mp4",
        "data/test_sample4.mp4",
        "test_sample5.mp4",
        "test_sample4.mp4"
    ]
    
    video_path = None
    for path in test_videos:
        if os.path.exists(path):
            video_path = path
            break
    
    if not video_path:
        print("❌ No test video found. Please ensure a test video is available.")
        print("   Expected locations: data/test_sample5.mp4 or test_sample5.mp4")
        return
    
    # Test video upload
    session_id = test_upload_video(video_path)
    if not session_id:
        return
    
    print()
    
    # Wait for processing to complete
    print("⏳ Waiting for video processing to complete...")
    time.sleep(5)
    
    # Test session info
    test_session_info(session_id)
    print()
    
    # Test chat
    test_messages = [
        "What is this video about?",
        "moments",
        "Can you summarize the key points?"
    ]
    
    for message in test_messages:
        test_chat(session_id, message)
        time.sleep(1)
        print()
    
    # Test moments
    test_moments(session_id)
    
    print()
    print("🎉 API testing completed!")
    print("💡 To test the Gradio interface, run 'python gradio_app.py'")
    print("🚀 To run both interfaces, use 'python server.py'")

if __name__ == "__main__":
    main()