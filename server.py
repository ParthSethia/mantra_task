import os
import sys
import threading
import time
from flask import Flask
import gradio as gr

# Add testing and tools directories to path
sys.path.append('testing')
sys.path.append('tools')

# Import both applications
from app import app as flask_app, chat_sessions
from gradio_app import demo as gradio_demo

def run_flask():
    """Run Flask API server"""
    print("🔧 Starting Flask API Server on http://localhost:5000")
    flask_app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

def run_gradio():
    """Run Gradio interface"""
    print("🎨 Starting Gradio Interface on http://localhost:7860")
    gradio_demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,
        share=False,
        prevent_thread_lock=True,
        debug=True
    )

def main():
    """Run both Flask and Gradio servers"""
    print("=" * 60)
    print("🚀 MULTIMODAL VIDEO CHAT ASSISTANT")
    print("=" * 60)
    print("")
    print("Starting hybrid server with both API and Web Interface...")
    print("")
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Give Flask time to start
    time.sleep(2)
    
    # Start Gradio in main thread
    gradio_demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,
        share=False
    )

if __name__ == "__main__":
    print("")
    print("📋 Available Interfaces:")
    print("  🌐 Web Interface: http://localhost:7860 (Gradio)")
    print("  🔌 API Server:    http://localhost:5000 (Flask)")
    print("")
    print("📖 API Documentation:")
    print("  POST /api/upload-video    - Upload and process video")
    print("  POST /api/chat           - Send message to chatbot")
    print("  GET  /api/session/<id>   - Get session info")
    print("  GET  /health             - Health check")
    print("")
    print("🎯 Features:")
    print("  • Video processing with transcription and frame analysis")
    print("  • Object detection and tracking with persistent IDs")
    print("  • Temporal reasoning and context-aware conversations")
    print("  • Support for CCTV/surveillance footage")
    print("  • Multi-turn chat with memory and context retention")
    print("")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")