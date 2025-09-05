import os
import sys
import tempfile
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import yaml

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from tools.enhanced_chatbot import EnhancedVideoChat

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store active chat sessions
chat_sessions = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Video Chat API'})

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """Upload and process a new video"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported: mp4, avi, mov, mkv, webm'}), 400
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        
        # Use single config - auto-detection handled by video processor
        config_path = request.form.get('config_path', 'config.yaml')
        
        # Initialize chat session
        try:
            chat = EnhancedVideoChat(config_path)
            result = chat.process_new_video(file_path)
            
            # Store session
            chat_sessions[session_id] = {
                'chat': chat,
                'video_path': file_path,
                'filename': filename,
                'config_path': config_path
            }
            
            return jsonify({
                'session_id': session_id,
                'message': result,
                'filename': filename,
                'status': 'success'
            })
            
        except Exception as e:
            # Clean up file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Failed to process video: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Send a message to the chatbot"""
    try:
        data = request.json
        session_id = data.get('session_id')
        message = data.get('message')
        
        if not session_id or not message:
            return jsonify({'error': 'session_id and message are required'}), 400
        
        if session_id not in chat_sessions:
            return jsonify({'error': 'Invalid session_id or session expired'}), 404
        
        chat_instance = chat_sessions[session_id]['chat']
        response = chat_instance.chat(message)
        
        return jsonify({
            'response': response,
            'session_id': session_id,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session_info(session_id):
    """Get information about a chat session"""
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session_info = chat_sessions[session_id]
        
        return jsonify({
            'session_id': session_id,
            'filename': session_info['filename'],
            'config_used': session_info['config_path'],
            'status': 'active'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get session info: {str(e)}'}), 500

@app.route('/api/session/<session_id>/moments', methods=['GET'])
def get_important_moments(session_id):
    """Get important moments from the video"""
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        chat_instance = chat_sessions[session_id]['chat']
        moments = chat_instance.list_important_moments()
        
        return jsonify({
            'moments': moments,
            'session_id': session_id,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get moments: {str(e)}'}), 500

@app.route('/api/session/<session_id>/objects', methods=['GET'])
def get_objects(session_id):
    """Get detected objects from the video"""
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        chat_instance = chat_sessions[session_id]['chat']
        objects = chat_instance.get_object_information("list all objects")
        
        return jsonify({
            'objects': objects,
            'session_id': session_id,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get objects: {str(e)}'}), 500

@app.route('/api/session/<session_id>/transcript', methods=['GET'])
def get_transcript(session_id):
    """Get video transcript"""
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        chat_instance = chat_sessions[session_id]['chat']
        
        # Get transcript from video data
        if chat_instance.current_video_data and 'transcript' in chat_instance.current_video_data:
            transcript = chat_instance.current_video_data['transcript']
            return jsonify({
                'transcript': transcript,
                'session_id': session_id,
                'status': 'success'
            })
        else:
            return jsonify({'error': 'Transcript not available'}), 404
        
    except Exception as e:
        return jsonify({'error': f'Failed to get transcript: {str(e)}'}), 500

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session and cleanup resources"""
    try:
        if session_id not in chat_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        # Get session info
        session_info = chat_sessions[session_id]
        
        # Clean up uploaded file
        if os.path.exists(session_info['video_path']):
            os.remove(session_info['video_path'])
        
        # Remove session
        del chat_sessions[session_id]
        
        return jsonify({
            'message': 'Session deleted successfully',
            'session_id': session_id,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to delete session: {str(e)}'}), 500

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    try:
        sessions = []
        for session_id, session_info in chat_sessions.items():
            sessions.append({
                'session_id': session_id,
                'filename': session_info['filename'],
                'config_used': session_info['config_path']
            })
        
        return jsonify({
            'sessions': sessions,
            'total': len(sessions),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to list sessions: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Video Chat API Server...")
    print("📝 API Documentation:")
    print("  POST /api/upload-video - Upload and process video")
    print("  POST /api/chat - Send message to chatbot")
    print("  GET  /api/session/<id> - Get session info")
    print("  GET  /api/session/<id>/moments - Get important moments")
    print("  GET  /api/session/<id>/objects - Get detected objects")
    print("  GET  /api/session/<id>/transcript - Get video transcript")
    print("  DELETE /api/session/<id> - Delete session")
    print("  GET  /api/sessions - List all sessions")
    print("  GET  /health - Health check")
    print("\n💡 Server running at http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)