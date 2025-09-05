import os
import sys
from pathlib import Path
import gradio as gr
import json

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from tools.enhanced_chatbot import EnhancedVideoChat

class GradioVideoChat:
    def __init__(self):
        self.chat_instances = {}
        self.current_session = None
        
    def process_video(self, video_file):
        """Process uploaded video and initialize chat"""
        if video_file is None:
            return "❌ Please upload a video file first.", "", []
        
        try:
            # Use single config - auto-detection handled by video processor
            config_path = "config.yaml"
            
            # Create chat instance
            chat = EnhancedVideoChat(config_path)
            
            # Process video
            result = chat.process_new_video(video_file)
            
            # Store session
            session_id = f"gradio_{id(chat)}"
            self.chat_instances[session_id] = chat
            self.current_session = session_id
            
            # Create initial summary
            summary = f"✅ **Video Processed Successfully!**\n\n"
            summary += f"📝 **Processing Result:**\n{result}\n\n"
            summary += "💬 **Ready for your questions!**"
            
            return summary, session_id, []
            
        except Exception as e:
            error_msg = f"❌ **Error Processing Video:**\n{str(e)}\n\n"
            error_msg += "💡 **Troubleshooting Tips:**\n"
            error_msg += "• Ensure video format is supported (mp4, avi, mov, mkv)\n"
            error_msg += "• Check if all dependencies are installed\n"
            error_msg += "• Try with a smaller video file\n"
            error_msg += "• Verify Ollama is running for VLM processing"
            return error_msg, "", []
    
    def chat_with_video(self, message, chat_history, session_id):
        """Handle chat interaction with the video"""
        if not message.strip():
            return chat_history, ""
        
        if not session_id or session_id not in self.chat_instances:
            error_response = "❌ No active video session. Please upload and process a video first."
            chat_history.append([message, error_response])
            return chat_history, ""
        
        try:
            chat_instance = self.chat_instances[session_id]
            
            # Handle special commands
            if message.lower() == "moments":
                response = chat_instance.list_important_moments()
            elif message.lower() == "objects":
                response = chat_instance.get_object_information("list all objects")
            elif message.lower() == "summary":
                # Get summary from video processor
                if hasattr(chat_instance, 'current_video_data') and chat_instance.current_video_data:
                    summary_data = chat_instance.current_video_data.get('processing_summary', {})
                    response = f"📊 **Video Summary:**\n{json.dumps(summary_data, indent=2)}"
                else:
                    response = "Summary not available"
            elif message.lower() == "transcript":
                # Get transcript
                if hasattr(chat_instance, 'current_video_data') and chat_instance.current_video_data:
                    transcript = chat_instance.current_video_data.get('transcript', {})
                    if transcript:
                        response = "📝 **Video Transcript:**\n"
                        for segment in transcript.get('segments', [])[:10]:  # First 10 segments
                            start_time = segment.get('start', 0)
                            text = segment.get('text', '')
                            response += f"[{start_time:.1f}s] {text}\n"
                        if len(transcript.get('segments', [])) > 10:
                            response += f"\n... and {len(transcript.get('segments', [])) - 10} more segments"
                    else:
                        response = "Transcript not available"
                else:
                    response = "Transcript not available"
            else:
                # Regular chat
                response = chat_instance.chat(message)
            
            chat_history.append([message, response])
            return chat_history, ""
            
        except Exception as e:
            error_response = f"❌ Error: {str(e)}"
            chat_history.append([message, error_response])
            return chat_history, ""
    
    def get_video_info(self, session_id):
        """Get information about the processed video"""
        if not session_id or session_id not in self.chat_instances:
            return "No active session"
        
        try:
            chat_instance = self.chat_instances[session_id]
            if hasattr(chat_instance, 'current_video_data') and chat_instance.current_video_data:
                info = "📊 **Video Information:**\n\n"
                
                # Basic info
                processing_summary = chat_instance.current_video_data.get('processing_summary', {})
                info += f"⏱️ **Duration:** {processing_summary.get('duration', 'Unknown')}\n"
                info += f"🖼️ **Total Frames:** {processing_summary.get('total_frames', 'Unknown')}\n"
                info += f"📝 **Transcript Segments:** {processing_summary.get('transcript_segments', 'Unknown')}\n"
                
                # Important moments
                important_moments = processing_summary.get('important_moments_count', 0)
                info += f"⭐ **Important Moments:** {important_moments}\n"
                
                # Objects detected
                objects_detected = processing_summary.get('objects_detected', 0)
                info += f"🎯 **Objects Detected:** {objects_detected}\n"
                
                return info
            else:
                return "Video information not available"
                
        except Exception as e:
            return f"Error getting video info: {str(e)}"

# Initialize the chat system
gradio_chat = GradioVideoChat()

# Create Gradio interface
with gr.Blocks(
    title="🎥 Multimodal Video Chat Assistant",
    theme=gr.themes.Soft(),
    css="""
        .container { max-width: 1200px; margin: auto; }
        .video-container { border-radius: 10px; }
        .chat-container { border-radius: 10px; }
        .info-box { background: rgb(16, 24, 39); padding: 15px; border-radius: 10px; color: white; }
    """
) as demo:
    
    # Header
    # gr.Markdown("""
    # # 🎥 Multimodal Video Chat Assistant
    
    # Upload a video and have intelligent conversations about its content! 
    
    # **Features:**
    # - 🎬 Video processing with audio transcription and frame analysis
    # - 🤖 Multi-turn conversations with context awareness  
    # - 🎯 Object detection and tracking
    # - ⏰ Temporal reasoning ("What happened before 2:30?")
    # - 📊 Automatic summarization and key moments identification
    # """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Video upload section
            gr.Markdown("### 📁 Upload Video")
            video_input = gr.Video(
                label="Upload Video File",
                elem_classes="video-container"
            )
            
            gr.Markdown("*Audio transcription will be automatically detected and skipped for silent videos*")
            
            process_btn = gr.Button(
                "🚀 Process Video",
                variant="primary",
                size="lg"
            )
            
            # Session info
            session_state = gr.Textbox(
                visible=False,
                label="Session ID"
            )
            
            # Video info display
            gr.Markdown("### 📊 Video Information")
            video_info = gr.Markdown(
                value="Upload and process a video to see information here.",
                elem_classes="info-box"
            )
        
        with gr.Column(scale=2):
            # Processing status
            processing_status = gr.Markdown(
                value="📤 **Ready to process video...**\n\nUpload a video file and click 'Process Video' to begin.",
                elem_classes="info-box"
            )
            
            # Chat interface
            gr.Markdown("### 💬 Chat with Your Video")
            
            chatbot = gr.Chatbot(
                label="Video Chat",
                height=500,
                elem_classes="chat-container",
                placeholder="Process a video to start chatting..."
            )
            
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask about your video... (try 'moments', 'objects', 'summary', 'transcript')",
                    label="Your Question",
                    scale=4
                )
                chat_btn = gr.Button("💬 Ask", scale=1)
            
            # Quick action buttons
            with gr.Row():
                moments_btn = gr.Button("⭐ Important Moments", size="sm")
                objects_btn = gr.Button("🎯 Objects", size="sm") 
                summary_btn = gr.Button("📋 Summary", size="sm")
                transcript_btn = gr.Button("📝 Transcript", size="sm")
    
    # Event handlers
    def update_video_info(session_id):
        return gradio_chat.get_video_info(session_id)
    
    # Process video
    process_btn.click(
        fn=gradio_chat.process_video,
        inputs=[video_input],
        outputs=[processing_status, session_state, chatbot]
    ).then(
        fn=update_video_info,
        inputs=[session_state],
        outputs=[video_info]
    )
    
    # Chat interactions
    def chat_and_clear(message, chat_history, session_id):
        updated_history, _ = gradio_chat.chat_with_video(message, chat_history, session_id)
        return updated_history, ""
    
    chat_btn.click(
        fn=chat_and_clear,
        inputs=[chat_input, chatbot, session_state],
        outputs=[chatbot, chat_input]
    )
    
    chat_input.submit(
        fn=chat_and_clear,
        inputs=[chat_input, chatbot, session_state],
        outputs=[chatbot, chat_input]
    )
    
    # Quick action buttons
    moments_btn.click(
        fn=lambda history, session_id: gradio_chat.chat_with_video("moments", history, session_id)[0],
        inputs=[chatbot, session_state],
        outputs=[chatbot]
    )
    
    objects_btn.click(
        fn=lambda history, session_id: gradio_chat.chat_with_video("objects", history, session_id)[0],
        inputs=[chatbot, session_state],
        outputs=[chatbot]
    )
    
    summary_btn.click(
        fn=lambda history, session_id: gradio_chat.chat_with_video("summary", history, session_id)[0],
        inputs=[chatbot, session_state],
        outputs=[chatbot]
    )
    
    transcript_btn.click(
        fn=lambda history, session_id: gradio_chat.chat_with_video("transcript", history, session_id)[0],
        inputs=[chatbot, session_state],
        outputs=[chatbot]
    )
    
    # Footer
    gr.Markdown("""
    ---
    ### 💡 Tips:
    - **Special Commands:** Type 'moments', 'objects', 'summary', or 'transcript' for quick info
    - **Temporal Queries:** Ask "What happened at 2:30?" or "Show me events before 1:45"
    - **Object References:** Ask about specific objects like "Where did car_1 go?"
    - **Context Aware:** The assistant remembers your conversation history
    
    **Supported Formats:** MP4, AVI, MOV, MKV, WEBM
    """)

if __name__ == "__main__":
    print("🚀 Starting Gradio Video Chat Interface...")
    print("📱 Interface will be available at: http://localhost:7860")
    print("🔧 Features: Video processing, Object tracking, Temporal reasoning")
    
    # Launch with public sharing disabled by default for security
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,
        share=False  # Set to True if you want public sharing
    )