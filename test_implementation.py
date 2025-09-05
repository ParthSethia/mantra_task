#!/usr/bin/env python3
"""
Test script for the multimodal video chat assistant implementation
"""

import os
import sys
sys.path.append('tools')

from tools.enhanced_chatbot import EnhancedVideoChat

def test_video_processing():
    """Test basic video processing functionality"""
    print("="*60)
    print("TESTING MULTIMODAL VIDEO CHAT ASSISTANT")
    print("="*60)
    
    # Initialize chatbot
    try:
        chat_bot = EnhancedVideoChat()
        print("✓ ChatBot initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize ChatBot: {e}")
        return False
    
    # Check if test videos exist
    regular_video = "data/test_sample5.mp4"  # Regular video with audio
    cctv_video = "data/test_sample4.mp4"     # CCTV footage
    
    test_video = cctv_video if os.path.exists(cctv_video) else regular_video
    
    if not os.path.exists(test_video):
        print(f"✗ No test video found. Tried:")
        print(f"  - CCTV footage: {cctv_video}")
        print(f"  - Regular video: {regular_video}")
        print("Please ensure at least one test video is available.")
        return False
    
    video_type = "CCTV/Surveillance" if test_video == cctv_video else "Regular/Audio"
    print(f"✓ Test video found: {test_video} ({video_type})")
    
    # Use appropriate config for video type
    config_path = "config_cctv.yaml" if test_video == cctv_video else "config.yaml"
    if test_video == cctv_video and os.path.exists("config_cctv.yaml"):
        print("✓ Using CCTV-optimized configuration")
        chat_bot = EnhancedVideoChat(config_path)
    else:
        print("✓ Using standard configuration")
        # chat_bot already initialized above
    
    # Process video
    try:
        print("\n" + "="*40)
        print("PROCESSING VIDEO...")
        print("="*40)
        
        # First processing (should be full processing)
        print("🔄 First processing (will process from scratch)...")
        summary = chat_bot.process_new_video(test_video)
        
        print("\n" + "="*40)
        print("VIDEO PROCESSING COMPLETE!")
        print("="*40)
        print("\nAUTO-GENERATED SUMMARY:")
        print("-" * 40)
        print(summary)
        
        # Test cache by processing the same video again
        print("\n" + "="*40)
        print("TESTING CACHE FUNCTIONALITY")
        print("="*40)
        print("🚀 Processing same video again (should load from cache)...")
        
        # Initialize a new chatbot instance to test cache loading
        cache_test_bot = EnhancedVideoChat(config_path) if test_video == cctv_video and os.path.exists("config_cctv.yaml") else EnhancedVideoChat()
        cached_summary = cache_test_bot.process_new_video(test_video)
        
        print("✓ Cache test completed!")
        
    except Exception as e:
        print(f"✗ Video processing failed: {e}")
        return False
    
    # Test chat functionality
    try:
        print("\n" + "="*40)
        print("TESTING CHAT FUNCTIONALITY")
        print("="*40)
        
        # Customize questions based on video type
        if test_video == cctv_video:
            test_questions = [
                "What type of surveillance footage is this?",
                "What objects were detected in this video?",
                "What are the key visual events?",
                "Can you list important timestamps with activities?",
                "What happened before 1:30?"  # Test temporal queries
            ]
            print("📹 Testing CCTV/Surveillance-specific questions...")
        else:
            test_questions = [
                "What is this video about?",
                "What are the key moments in this video?",
                "Can you list the important timestamps?",
                "What was said in the video?",
                "What happened before 2:00?"  # Test temporal queries
            ]
            print("🎬 Testing regular video questions...")
        
        for question in test_questions:
            print(f"\nQ: {question}")
            print("-" * 20)
            response = chat_bot.chat(question)
            print(f"A: {response[:200]}..." if len(response) > 200 else f"A: {response}")
            
    except Exception as e:
        print(f"✗ Chat functionality failed: {e}")
        return False
    
    print("\n" + "="*40)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*40)
    print(f"\nThe video chat assistant is ready for use with {video_type} content!")
    
    if test_video == cctv_video:
        print("🔍 CCTV/Surveillance Features Available:")
        print("1. Object detection and tracking (person, car, etc.)")
        print("2. Visual event analysis without audio")
        print("3. Temporal queries ('What happened before 2:30?')")
        print("4. Activity monitoring and movement tracking")
        print("5. Security-focused summarization")
    else:
        print("🎬 Regular Video Features Available:")
        print("1. Audio + visual content analysis")
        print("2. Transcript-based queries")
        print("3. Multi-modal understanding")
        print("4. Educational/entertainment content processing")
        print("5. Speech and visual correlation")
    
    print("\n🚀 Advanced Features:")
    print("- Intelligent caching (instant reprocessing)")
    print("- Graph-based temporal reasoning")
    print("- Multi-turn conversational memory")
    print("- Object-aware conversations")
    print("- VLM model flexibility (Ollama/BLIP/Florence-2)")
    
    return True

def interactive_chat_demo():
    """Run an interactive chat demo"""
    print("\n" + "="*40)
    print("INTERACTIVE CHAT DEMO")
    print("="*40)
    
    # Determine video type and config
    cctv_video = "data/test_sample4.mp4"
    regular_video = "data/test_sample5.mp4"
    
    if os.path.exists(cctv_video):
        video_path = cctv_video
        video_type = "CCTV/Surveillance"
        config_path = "config_cctv.yaml" if os.path.exists("config_cctv.yaml") else "config.yaml"
        chat_bot = EnhancedVideoChat(config_path)
        
        print("📹 CCTV Demo Mode")
        print("Special commands: 'objects', 'tracks', 'moments'")
        print("Try: 'What objects were detected?', 'What happened before 1:30?'")
        
    elif os.path.exists(regular_video):
        video_path = regular_video  
        video_type = "Regular/Audio"
        chat_bot = EnhancedVideoChat()
        
        print("🎬 Regular Video Demo Mode") 
        print("Special commands: 'moments', 'transcript'")
        print("Try: 'What was said?', 'What happened before 2:00?'")
    else:
        print("No test videos found for demo")
        return
    
    print(f"Video: {video_path} ({video_type})")
    print("Type 'quit' to exit")
    print("="*40)
    
    if os.path.exists(video_path):
        # Process video first (will use cache if available)
        print("Loading video (checking cache first)...")
        chat_bot.process_new_video(video_path)
        
        print("\n✓ Ready for chat! Ask me anything about the video.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'moments':
                    response = chat_bot.list_important_moments()
                    print(f"\nAssistant: {response}")
                elif user_input.lower() == 'objects' and video_type == "CCTV/Surveillance":
                    response = chat_bot.get_object_information("What objects were detected?")
                    print(f"\nAssistant: {response}")
                elif user_input.lower() == 'tracks' and video_type == "CCTV/Surveillance":
                    response = chat_bot.get_object_information("Show me all tracks")
                    print(f"\nAssistant: {response}")
                elif user_input.lower() == 'transcript' and video_type == "Regular/Audio":
                    # Try to get transcript info
                    if chat_bot.current_video_data and chat_bot.current_video_data.get('transcript', {}).get('has_transcript'):
                        transcript = chat_bot.current_video_data['transcript']['full_text'][:500]
                        print(f"\nAssistant: Transcript preview: {transcript}...")
                    else:
                        print(f"\nAssistant: No transcript available for this video.")
                elif user_input:
                    response = chat_bot.chat(user_input)
                    print(f"\nAssistant: {response}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    else:
        print(f"Test video not found: {video_path}")

if __name__ == "__main__":
    # Run basic tests
    success = test_video_processing()
    
    if success:
        # Show cache status
        try:
            from tools.utils.video_cache import VideoCacheManager
            cache_manager = VideoCacheManager()
            cache_stats = cache_manager.get_cache_stats()
            
            print(f"\n📊 Cache Status:")
            print(f"   Entries: {cache_stats['valid_entries']}/{cache_stats['total_entries']}")
            print(f"   Size: {cache_stats['total_size_mb']:.1f} MB")
            print(f"   Method: {cache_stats['validation_method']}")
        except Exception:
            pass  # Skip cache status if error
        
        # Ask if user wants interactive demo
        try:
            demo = input("\nWould you like to try the interactive chat demo? (y/n): ").strip().lower()
            if demo in ['y', 'yes']:
                interactive_chat_demo()
        except KeyboardInterrupt:
            print("\nGoodbye!")
    else:
        print("\n✗ Tests failed. Please check the errors above.")
        sys.exit(1)