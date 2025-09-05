#!/usr/bin/env python3
"""
Test script for frame analysis system
Tests frame extraction, VLM analysis, and error handling
"""
import os
import sys
import json
import time
from pathlib import Path

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from tools.video_frame_processor import VideoFrameProcessor

class FrameAnalysisTester:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.processor = None
        
    def setup_processor(self):
        """Initialize the frame processor"""
        try:
            print("🔧 Initializing VideoFrameProcessor...")
            self.processor = VideoFrameProcessor(self.config_path)
            print("✅ VideoFrameProcessor initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize VideoFrameProcessor: {e}")
            return False
    
    def test_frame_extraction(self, video_path: str) -> bool:
        """Test basic frame extraction"""
        print(f"\n📹 Testing frame extraction for: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return False
        
        try:
            frames_data = self.processor.extract_frames(video_path)
            
            if not frames_data:
                print("❌ No frames extracted")
                return False
            
            print(f"✅ Extracted {len(frames_data)} frames")
            
            # Validate first frame
            first_frame = frames_data[0]
            required_keys = ['frame_id', 'timestamp', 'frame_path', 'pil_image', 'base64_image']
            missing_keys = [key for key in required_keys if key not in first_frame]
            
            if missing_keys:
                print(f"❌ Missing keys in frame data: {missing_keys}")
                return False
            
            # Check if frame file exists
            if not os.path.exists(first_frame['frame_path']):
                print(f"❌ Frame file not saved: {first_frame['frame_path']}")
                return False
            
            # Check PIL image
            if first_frame['pil_image'] is None:
                print("❌ PIL image is None")
                return False
            
            # Check base64 image
            if not first_frame['base64_image'] or len(first_frame['base64_image']) < 100:
                print("❌ Base64 image is empty or too small")
                return False
            
            print("✅ Frame extraction validation passed")
            print(f"   First frame: {first_frame['frame_id']} at {first_frame['timestamp']:.2f}s")
            print(f"   PIL image size: {first_frame['pil_image'].size}")
            print(f"   Base64 length: {len(first_frame['base64_image'])} chars")
            
            return True
            
        except Exception as e:
            print(f"❌ Frame extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_single_frame_analysis(self, frames_data: list) -> bool:
        """Test single frame VLM analysis"""
        print(f"\n🤖 Testing single frame VLM analysis...")
        
        if not frames_data:
            print("❌ No frames data provided")
            return False
        
        try:
            # Test first frame
            first_frame = frames_data[0]
            print(f"   Analyzing frame {first_frame['frame_id']} at {first_frame['timestamp']:.2f}s")
            
            result = self.processor.analyze_frame_importance_single(first_frame)
            
            # Validate result structure
            required_keys = ['frame_id', 'timestamp', 'frame_path', 'vlm_analysis', 'importance_score', 'analyzed']
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                print(f"❌ Missing keys in analysis result: {missing_keys}")
                return False
            
            print("✅ Single frame analysis completed")
            print(f"   Analysis: {result['vlm_analysis'][:100]}...")
            print(f"   Importance score: {result['importance_score']}")
            print(f"   Analyzed: {result['analyzed']}")
            
            # Check for error patterns
            analysis_text = result['vlm_analysis']
            if "Analysis failed" in analysis_text:
                print(f"⚠️  Analysis contains failure message: {analysis_text}")
                return False
            
            if "'NoneType'" in analysis_text and "shape" in analysis_text:
                print(f"❌ NoneType shape error detected: {analysis_text}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Single frame analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_batch_analysis(self, frames_data: list, batch_size: int = 3) -> bool:
        """Test batch frame analysis"""
        print(f"\n🔄 Testing batch frame analysis (batch_size={batch_size})...")
        
        if len(frames_data) < batch_size:
            print(f"❌ Not enough frames for batch analysis: {len(frames_data)} < {batch_size}")
            return False
        
        try:
            # Test first batch
            batch = frames_data[:batch_size]
            print(f"   Analyzing batch of {len(batch)} frames")
            
            results = self.processor.analyze_frames_batch(batch)
            
            if len(results) != len(batch):
                print(f"❌ Batch analysis returned {len(results)} results for {len(batch)} frames")
                return False
            
            # Validate each result
            failed_analyses = 0
            for i, result in enumerate(results):
                analysis_text = result.get('vlm_analysis', '')
                if "Analysis failed" in analysis_text or "'NoneType'" in analysis_text:
                    failed_analyses += 1
                    print(f"   ⚠️  Frame {i} analysis failed: {analysis_text[:50]}...")
            
            success_rate = (len(results) - failed_analyses) / len(results) * 100
            print(f"✅ Batch analysis completed")
            print(f"   Success rate: {success_rate:.1f}% ({len(results) - failed_analyses}/{len(results)})")
            
            return failed_analyses == 0  # Return True only if all analyses succeeded
            
        except Exception as e:
            print(f"❌ Batch analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_full_video_processing(self, video_path: str) -> bool:
        """Test complete video frame processing pipeline"""
        print(f"\n🎬 Testing full video processing pipeline...")
        
        try:
            start_time = time.time()
            analyzed_frames = self.processor.process_video_frames(video_path)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if not analyzed_frames:
                print("❌ No analyzed frames returned")
                return False
            
            # Analyze results
            total_frames = len(analyzed_frames)
            failed_frames = sum(1 for frame in analyzed_frames 
                              if "Analysis failed" in frame.get('vlm_analysis', '') 
                              or "'NoneType'" in frame.get('vlm_analysis', ''))
            
            success_frames = total_frames - failed_frames
            success_rate = success_frames / total_frames * 100 if total_frames > 0 else 0
            
            print(f"✅ Full video processing completed")
            print(f"   Total frames: {total_frames}")
            print(f"   Successful analyses: {success_frames}")
            print(f"   Failed analyses: {failed_frames}")
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Processing time: {processing_time:.2f} seconds")
            print(f"   Average time per frame: {processing_time/total_frames:.2f}s")
            
            # Get important frames
            important_frames = self.processor.get_important_frames(analyzed_frames)
            print(f"   Important frames: {len(important_frames)}")
            
            return success_rate > 80  # Pass if >80% success rate
            
        except Exception as e:
            print(f"❌ Full video processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_error_handling(self, video_path: str) -> bool:
        """Test error handling with various scenarios"""
        print(f"\n🛡️  Testing error handling...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Non-existent video
        total_tests += 1
        try:
            fake_path = "/nonexistent/video.mp4"
            frames = self.processor.extract_frames(fake_path)
            if not frames:  # Should return empty list gracefully
                print("✅ Non-existent video handled gracefully")
                tests_passed += 1
            else:
                print("❌ Non-existent video should return empty frames")
        except Exception as e:
            print(f"⚠️  Non-existent video threw exception: {e}")
        
        # Test 2: None PIL image handling
        total_tests += 1
        try:
            fake_frame = {
                'frame_id': 999,
                'timestamp': 0.0,
                'frame_path': '/fake/path.jpg',
                'pil_image': None,
                'base64_image': None
            }
            result = self.processor.analyze_frame_importance_single(fake_frame)
            if "Analysis failed" in result.get('vlm_analysis', ''):
                print("✅ None PIL image handled gracefully")
                tests_passed += 1
            else:
                print("❌ None PIL image should fail gracefully")
        except Exception as e:
            print(f"⚠️  None PIL image threw exception: {e}")
        
        success_rate = tests_passed / total_tests * 100 if total_tests > 0 else 0
        print(f"   Error handling success rate: {success_rate:.1f}% ({tests_passed}/{total_tests})")
        
        return success_rate >= 50  # Pass if at least half the error tests pass
    
    def run_full_test_suite(self, video_path: str):
        """Run complete test suite"""
        print("=" * 70)
        print("🧪 FRAME ANALYSIS TEST SUITE")
        print("=" * 70)
        
        # Setup
        if not self.setup_processor():
            print("❌ Test suite failed - could not initialize processor")
            return False
        
        # Find test video
        test_videos = [video_path] if video_path else [
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
        
        # Run tests
        test_results = {}
        
        # Test 1: Frame Extraction
        test_results['extraction'] = self.test_frame_extraction(video_file)
        
        if test_results['extraction']:
            # Extract frames for subsequent tests
            frames_data = self.processor.extract_frames(video_file)
            
            # Test 2: Single Frame Analysis
            test_results['single_analysis'] = self.test_single_frame_analysis(frames_data)
            
            # Test 3: Batch Analysis
            if len(frames_data) >= 3:
                test_results['batch_analysis'] = self.test_batch_analysis(frames_data)
            else:
                test_results['batch_analysis'] = True  # Skip if not enough frames
                print("⏭️  Skipping batch analysis - not enough frames")
        
        # Test 4: Full Pipeline
        test_results['full_pipeline'] = self.test_full_video_processing(video_file)
        
        # Test 5: Error Handling
        test_results['error_handling'] = self.test_error_handling(video_file)
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 70)
        
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():.<50} {status}")
        
        overall_success = passed_tests / total_tests * 100
        print(f"\nOverall Success Rate: {overall_success:.1f}% ({passed_tests}/{total_tests})")
        
        if overall_success >= 80:
            print("🎉 Frame analysis system is working well!")
        elif overall_success >= 50:
            print("⚠️  Frame analysis system has some issues but is functional")
        else:
            print("❌ Frame analysis system needs significant fixes")
        
        return overall_success >= 80

def main():
    """Run the frame analysis tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test frame analysis system')
    parser.add_argument('--video', '-v', type=str, help='Path to test video file')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--quick', '-q', action='store_true', help='Run quick tests only')
    
    args = parser.parse_args()
    
    tester = FrameAnalysisTester(args.config)
    
    if args.quick:
        # Quick test - just setup and basic extraction
        print("🚀 Running quick frame analysis test...")
        if tester.setup_processor():
            video_file = args.video or "data/test_sample5.mp4"
            if os.path.exists(video_file):
                success = tester.test_frame_extraction(video_file)
                print(f"\n{'✅ Quick test PASSED' if success else '❌ Quick test FAILED'}")
            else:
                print(f"❌ Test video not found: {video_file}")
        else:
            print("❌ Quick test failed - processor setup failed")
    else:
        # Full test suite
        success = tester.run_full_test_suite(args.video)
        exit(0 if success else 1)

if __name__ == "__main__":
    main()