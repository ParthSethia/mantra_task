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
    
    # Check if test video exists
    video_path = "data/test_sample5.mp4"
    if not os.path.exists(video_path):
        print(f"✗ Test video not found: {video_path}")
        print("Please ensure the test video is available at the configured path.")
        return False
    
    print(f"✓ Test video found: {video_path}")
    
    # Process video
    try:
        print("\n" + "="*40)
        print("PROCESSING VIDEO...")
        print("="*40)
        
        summary = chat_bot.process_new_video(video_path)
        
        print("\n" + "="*40)
        print("VIDEO PROCESSING COMPLETE!")
        print("="*40)
        print("\nAUTO-GENERATED SUMMARY:")
        print("-" * 40)
        print(summary)
        
    except Exception as e:
        print(f"✗ Video processing failed: {e}")
        return False
    
    # Test chat functionality
    try:
        print("\n" + "="*40)
        print("TESTING CHAT FUNCTIONALITY")
        print("="*40)
        
        test_questions = [
            "What is this video about?",
            "What are the key moments in this video?",
            "Can you list the important timestamps?"
        ]
        
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
    print("\nThe video chat assistant is ready for use!")
    print("You can now:")
    print("1. Ask questions about the video content")
    print("2. Request specific timestamps")
    print("3. Get detailed context around moments")
    print("4. Search for specific topics or events")
    
    return True

def interactive_chat_demo():
    """Run an interactive chat demo"""
    print("\n" + "="*40)
    print("INTERACTIVE CHAT DEMO")
    print("="*40)
    print("Type 'quit' to exit, 'moments' to see important moments")
    
    chat_bot = EnhancedVideoChat()
    video_path = "data/test_sample5.mp4"
    
    if os.path.exists(video_path):
        # Process video first
        print("Processing video for demo...")
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