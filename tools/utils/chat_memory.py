import yaml
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class ChatMemory:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.memory_config = config['memory']
        self.cache_config = config['cache']
        
        self.session_file = os.path.join(self.cache_config['metadata_directory'], 'chat_sessions.json')
        self.current_session = None
        
        # Create metadata directory
        os.makedirs(self.cache_config['metadata_directory'], exist_ok=True)
        
        # Initialize session storage
        if not os.path.exists(self.session_file):
            self._save_sessions({})
    
    def start_new_session(self, session_id: str = None) -> str:
        """Start a new chat session"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'messages': [],
            'video_context': None,
            'active_timestamp': None,
            'mentioned_objects': [],
            'conversation_summary': ""
        }
        
        return session_id
    
    def add_message(self, message: str, message_type: str = 'user', metadata: Dict = None):
        """Add a message to current session"""
        if not self.current_session:
            self.start_new_session()
        
        message_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': message_type,  # 'user' or 'assistant'
            'content': message,
            'metadata': metadata or {}
        }
        
        self.current_session['messages'].append(message_entry)
        
        # Keep only last N messages
        max_history = self.memory_config['max_conversation_history']
        if len(self.current_session['messages']) > max_history:
            self.current_session['messages'] = self.current_session['messages'][-max_history:]
    
    def set_video_context(self, video_path: str, video_metadata: Dict = None):
        """Set the video context for current session"""
        if not self.current_session:
            self.start_new_session()
        
        self.current_session['video_context'] = {
            'video_path': video_path,
            'metadata': video_metadata,
            'set_at': datetime.now().isoformat()
        }
    
    def set_active_timestamp(self, timestamp: float):
        """Set the currently discussed timestamp"""
        if not self.current_session:
            self.start_new_session()
        
        self.current_session['active_timestamp'] = timestamp
    
    def add_mentioned_object(self, object_id: str, object_type: str, timestamp: float):
        """Track mentioned objects in conversation"""
        if not self.current_session:
            self.start_new_session()
        
        object_mention = {
            'object_id': object_id,
            'object_type': object_type,
            'timestamp': timestamp,
            'mentioned_at': datetime.now().isoformat()
        }
        
        # Avoid duplicates
        existing = [obj for obj in self.current_session['mentioned_objects'] 
                   if obj['object_id'] == object_id and obj['timestamp'] == timestamp]
        
        if not existing:
            self.current_session['mentioned_objects'].append(object_mention)
    
    def get_conversation_context(self, last_n_messages: int = 10) -> Dict:
        """Get recent conversation context"""
        if not self.current_session:
            return {'messages': [], 'video_context': None}
        
        recent_messages = self.current_session['messages'][-last_n_messages:]
        
        return {
            'session_id': self.current_session['session_id'],
            'messages': recent_messages,
            'video_context': self.current_session['video_context'],
            'active_timestamp': self.current_session.get('active_timestamp'),
            'mentioned_objects': self.current_session.get('mentioned_objects', [])
        }
    
    def save_current_session(self):
        """Save current session to persistent storage"""
        if not self.current_session:
            return
        
        sessions = self._load_sessions()
        sessions[self.current_session['session_id']] = self.current_session.copy()
        self._save_sessions(sessions)
    
    def load_session(self, session_id: str) -> bool:
        """Load a previous session"""
        sessions = self._load_sessions()
        if session_id in sessions:
            self.current_session = sessions[session_id]
            return True
        return False
    
    def get_session_history(self) -> List[Dict]:
        """Get list of all sessions"""
        sessions = self._load_sessions()
        return [
            {
                'session_id': sid,
                'start_time': session_data.get('start_time'),
                'message_count': len(session_data.get('messages', [])),
                'video_context': session_data.get('video_context', {}).get('video_path')
            }
            for sid, session_data in sessions.items()
        ]
    
    def cleanup_old_sessions(self):
        """Remove sessions older than retention period"""
        if not self.memory_config['enable_cross_session_memory']:
            return
        
        sessions = self._load_sessions()
        retention_hours = self.memory_config['session_retention_hours']
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        sessions_to_keep = {}
        for sid, session_data in sessions.items():
            start_time = datetime.fromisoformat(session_data.get('start_time', ''))
            if start_time > cutoff_time:
                sessions_to_keep[sid] = session_data
        
        self._save_sessions(sessions_to_keep)
    
    def _load_sessions(self) -> Dict:
        """Load sessions from file"""
        try:
            with open(self.session_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_sessions(self, sessions: Dict):
        """Save sessions to file"""
        with open(self.session_file, 'w') as f:
            json.dump(sessions, f, indent=2)


# Global instance
chat_memory = ChatMemory()