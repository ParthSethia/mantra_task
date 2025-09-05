#!/usr/bin/env python3
"""
Debug script to trace the exact cause of the NoneType shape error
"""
import os
import sys

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

def main():
    """Debug the VLM issue step by step"""
    print("🔍 DEBUGGING VLM ANALYSIS ISSUE")
    print("=" * 50)
    
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
        return
    
    print(f"📁 Using test video: {video_file}")
    
    # Step 1: Test frame extraction only
    print("\n🎬 Step 1: Testing frame extraction...")
    try:
        from tools.video_frame_processor import VideoFrameProcessor
        processor = VideoFrameProcessor('config.yaml')
        print(f"   Model type: {processor.model_type}")
        
        frames_data = processor.extract_frames(video_file)
        print(f"   Extracted {len(frames_data)} frames")
        
        if frames_data:
            first_frame = frames_data[0]
            print(f"   First frame keys: {list(first_frame.keys())}")
            print(f"   PIL image is None: {first_frame.get('pil_image') is None}")
            print(f"   Frame path exists: {os.path.exists(first_frame.get('frame_path', ''))}")
            
            if first_frame.get('pil_image'):
                print(f"   PIL image size: {first_frame['pil_image'].size}")
            
            # Step 2: Test single analysis with debugging
            print(f"\n🔬 Step 2: Testing single frame analysis with debugging...")
            result = processor.analyze_frame_importance_single(first_frame)
            
            print(f"   Analysis result keys: {list(result.keys())}")
            print(f"   Analysis text: {result.get('vlm_analysis', 'No analysis')[:200]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()