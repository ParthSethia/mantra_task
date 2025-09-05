#!/usr/bin/env python3
"""
Quick test script to verify VLM analysis fixes
Tests the different VLM model types separately
"""
import os
import sys
import yaml

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from tools.video_frame_processor import VideoFrameProcessor

def test_vlm_initialization():
    """Test VLM initialization for different model types"""
    print("🔧 Testing VLM initialization...")
    
    # Read config to see current model type
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    current_model_type = config['vlm']['model_type']
    print(f"   Current model type: {current_model_type}")
    
    try:
        processor = VideoFrameProcessor('config.yaml')
        print(f"✅ {current_model_type} VLM initialized successfully")
        
        # Check what was initialized
        if processor.model_type == 'ollama':
            print(f"   Ollama chain: {'✅ Initialized' if processor.chain else '❌ None'}")
            print(f"   HF VLM: {'❌ None (expected)' if processor.hf_vlm is None else '⚠️ Unexpected'}")
        else:
            print(f"   HF VLM: {'✅ Initialized' if processor.hf_vlm else '❌ None'}")
            print(f"   Ollama chain: {'❌ None (expected)' if processor.chain is None else '⚠️ Unexpected'}")
        
        return processor
        
    except Exception as e:
        print(f"❌ VLM initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_single_frame_analysis(processor, video_path):
    """Test single frame analysis"""
    print(f"\n🖼️ Testing single frame analysis...")
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        # Extract first frame
        frames_data = processor.extract_frames(video_path)
        if not frames_data:
            print("❌ No frames extracted")
            return False
        
        print(f"   Extracted {len(frames_data)} frames")
        
        # Analyze first frame
        first_frame = frames_data[0]
        print(f"   Analyzing frame {first_frame['frame_id']} at {first_frame['timestamp']:.2f}s")
        
        result = processor.analyze_frame_importance_single(first_frame)
        
        # Check for the invoke error
        analysis = result.get('vlm_analysis', '')
        if "'NoneType' object has no attribute 'invoke'" in analysis:
            print("❌ Still getting NoneType invoke error")
            return False
        elif "Analysis failed" in analysis and "invoke" in analysis:
            print(f"❌ Analysis failed with invoke-related error: {analysis}")
            return False
        elif "Analysis failed" in analysis:
            print(f"⚠️ Analysis failed but not invoke-related: {analysis[:100]}...")
            return False
        else:
            print("✅ Single frame analysis successful!")
            print(f"   Analysis: {analysis[:100]}...")
            print(f"   Importance score: {result.get('importance_score', 'N/A')}")
            return True
            
    except Exception as e:
        print(f"❌ Single frame analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run VLM fix verification"""
    print("=" * 60)
    print("🧪 VLM FIX VERIFICATION")
    print("=" * 60)
    
    # Test 1: VLM Initialization
    processor = test_vlm_initialization()
    if not processor:
        print("❌ Cannot proceed - VLM initialization failed")
        return False
    
    # Find test video
    test_videos = [
        "data/test_sample5.mp4",
        "data/test_sample4.mp4", 
        "test_sample5.mp4",
        "test_sample4.mp4"
    ]
    
    video_file = None
    for path in test_videos:
        if os.path.exists(path):
            video_file = path
            break
    
    if not video_file:
        print(f"❌ No test video found. Tried: {test_videos}")
        return False
    
    print(f"📁 Using test video: {video_file}")
    
    # Test 2: Single Frame Analysis
    success = test_single_frame_analysis(processor, video_file)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION RESULTS")
    print("=" * 60)
    
    if success:
        print("🎉 VLM fix verification PASSED!")
        print("   The 'NoneType invoke' error should be resolved.")
    else:
        print("❌ VLM fix verification FAILED!")
        print("   The 'NoneType invoke' error may still exist.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)