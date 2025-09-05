#!/usr/bin/env python3
"""
Multimodal Video Chat Assistant - Main Interface
Interactive chatbot for video analysis and conversation
"""

import os
import sys
import argparse
from pathlib import Path

# Add tools directory to path
sys.path.append('tools')

def detect_video_type(video_path):
    """Detect if video is CCTV/surveillance or regular video"""
    video_name = Path(video_path).name.lower()
    
    # CCTV indicators
    cctv_keywords = ['cctv', 'surveillance', 'security', 'cam', 'monitor']
    if any(keyword in video_name for keyword in cctv_keywords):
        return 'cctv'
    
    # Check if it's test_sample4 (known CCTV footage)
    if 'test_sample4' in video_name:
        return 'cctv'
    
    # Default to regular video
    return 'regular'

def get_config_path(video_type, custom_config=None):
    """Get appropriate configuration file"""
    if custom_config and os.path.exists(custom_config):
        return custom_config
    
    if video_type == 'cctv' and os.path.exists('config_cctv.yaml'):
        return 'config_cctv.yaml'
    
    return 'config.yaml'

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🎬 MULTIMODAL VIDEO CHAT ASSISTANT")
    print("=" * 60)
    print("Advanced video analysis with object tracking and temporal reasoning")
    print("Supports regular videos with audio and CCTV surveillance footage")
    print("=" * 60)

def print_video_info(video_path, video_type, config_path):
    """Print video and configuration information"""
    print(f"\n📹 Video: {video_path}")
    print(f"🎯 Type: {'CCTV/Surveillance' if video_type == 'cctv' else 'Regular/Audio'}")
    print(f"⚙️  Config: {config_path}")
    
    if video_type == 'cctv':
        print("\n🔍 CCTV Mode Features:")
        print("  • Object detection and tracking")
        print("  • Visual event analysis (no audio)")
        print("  • Temporal queries ('What happened before 2:30?')")
        print("  • Security-focused summarization")
    else:
        print("\n🎬 Regular Video Features:")
        print("  • Audio + visual content analysis")
        print("  • Transcript-based queries")
        print("  • Multi-modal understanding")
        print("  • Object tracking + speech correlation")

def print_help():
    """Print available commands"""
    print("\n💡 Available Commands:")
    print("  'help' or '?'     - Show this help")
    print("  'moments'         - List important moments")
    print("  'objects'         - Show detected objects (CCTV mode)")
    print("  'tracks'          - Show object tracks (CCTV mode)")
    print("  'transcript'      - Show transcript preview (regular video)")
    print("  'summary'         - Regenerate video summary")
    print("  'quit' or 'exit'  - Exit the chatbot")
    print("\n📝 Example Questions:")
    if os.path.exists('config_cctv.yaml'):
        print("  • What objects were detected in this video?")
        print("  • What happened before 1:30?")
        print("  • Show me all car movements")
        print("  • Describe the activity at timestamp 2:15")
    else:
        print("  • What is this video about?")
        print("  • What was said around 2:30?")
        print("  • What are the key moments?")
        print("  • Summarize the main topics discussed")

def initialize_chatbot(config_path):
    """Initialize the enhanced chatbot"""
    try:
        from tools.enhanced_chatbot import EnhancedVideoChat
        print("🔄 Initializing chatbot...")
        chatbot = EnhancedVideoChat(config_path)
        print("✅ Chatbot initialized successfully")
        return chatbot
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("\nTroubleshooting:")
        print("1. Check if required packages are installed:")
        print("   pip install -r requirements.txt")
        print("2. Ensure Ollama is running (if using Ollama models)")
        print("3. Check configuration file exists and is valid")
        return None

def process_video(chatbot, video_path):
    """Process the video and generate initial summary"""
    print(f"\n🔄 Processing video: {os.path.basename(video_path)}")
    print("This may take a few moments for the first run...")
    print("(Subsequent runs will use cache for faster loading)")
    
    try:
        summary = chatbot.process_new_video(video_path)
        
        print("\n" + "=" * 50)
        print("✅ VIDEO PROCESSING COMPLETE!")
        print("=" * 50)
        print("\n📋 AUTO-GENERATED SUMMARY:")
        print("-" * 30)
        print(summary)
        print("-" * 30)
        
        return True
        
    except Exception as e:
        print(f"❌ Video processing failed: {e}")
        print("\nPossible issues:")
        print("1. Video file not found or corrupted")
        print("2. Unsupported video format")
        print("3. Missing dependencies for audio/object processing")
        print("4. Configuration issues")
        return False

def interactive_chat(chatbot, video_type):
    """Run interactive chat session"""
    print(f"\n{'='*50}")
    print("💬 INTERACTIVE CHAT STARTED")
    print(f"{'='*50}")
    print("✨ Ready to answer questions about your video!")
    print("Type 'help' for available commands or ask any question.")
    print("Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input(f"\n{'You':<12}: ").strip()
            
            if not user_input:
                continue
                
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Video Chat Assistant!")
                break
                
            elif user_input.lower() in ['help', '?']:
                print_help()
                continue
                
            elif user_input.lower() == 'moments':
                print(f"\n{'Assistant':<12}: ", end="")
                response = chatbot.list_important_moments()
                print(response)
                continue
                
            elif user_input.lower() == 'objects' and video_type == 'cctv':
                print(f"\n{'Assistant':<12}: ", end="")
                response = chatbot.get_object_information("What objects were detected?")
                print(response)
                continue
                
            elif user_input.lower() == 'tracks' and video_type == 'cctv':
                print(f"\n{'Assistant':<12}: ", end="")
                response = chatbot.get_object_information("Show me all tracks")
                print(response)
                continue
                
            elif user_input.lower() == 'transcript' and video_type == 'regular':
                print(f"\n{'Assistant':<12}: ", end="")
                if (chatbot.current_video_data and 
                    chatbot.current_video_data.get('transcript', {}).get('has_transcript')):
                    transcript = chatbot.current_video_data['transcript']['full_text'][:500]
                    print(f"Transcript preview: {transcript}...")
                else:
                    print("No transcript available for this video.")
                continue
                
            elif user_input.lower() == 'summary':
                print(f"\n{'Assistant':<12}: ", end="")
                if hasattr(chatbot, 'current_video_summary'):
                    print(chatbot.current_video_summary)
                else:
                    print("No summary available. Please process a video first.")
                continue
            
            # Regular chat query
            print(f"\n{'Assistant':<12}: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing your question: {e}")
            print("Please try rephrasing your question or type 'help' for assistance.")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Multimodal Video Chat Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data/test_sample5.mp4                    # Regular video
  python main.py data/test_sample4.mp4                    # CCTV footage
  python main.py video.mp4 --config config_custom.yaml   # Custom config
  python main.py surveillance.mp4 --type cctv            # Force CCTV mode

Supported formats: mp4, avi, mov, mkv
        """
    )
    
    parser.add_argument(
        "video_path",
        help="Path to the video file to analyze"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=['auto', 'regular', 'cctv'],
        default='auto',
        help="Video type (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the banner display"
    )
    
    args = parser.parse_args()
    
    # Print banner
    if not args.no_banner:
        print_banner()
    
    # Validate video file
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        print("Please check the file path and try again.")
        sys.exit(1)
    
    # Detect video type
    if args.type == 'auto':
        video_type = detect_video_type(args.video_path)
    else:
        video_type = args.type
    
    # Get configuration
    config_path = get_config_path(video_type, args.config)
    
    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        print("Please ensure the configuration file exists.")
        sys.exit(1)
    
    # Print video information
    print_video_info(args.video_path, video_type, config_path)
    
    # Initialize chatbot
    chatbot = initialize_chatbot(config_path)
    if not chatbot:
        sys.exit(1)
    
    # Process video
    if not process_video(chatbot, args.video_path):
        sys.exit(1)
    
    # Start interactive chat
    interactive_chat(chatbot, video_type)

if __name__ == "__main__":
    main()