import json
import os
import yaml
from typing import Dict, List, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from video_processor import VideoProcessor
from utils.chat_memory import ChatMemory
from utils.vector_store import VideoVectorStore

class EnhancedVideoChat:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.video_processor = VideoProcessor(config_path)
        self.chat_memory = ChatMemory(config_path)
        self.vector_store = VideoVectorStore(config_path)
        
        # Initialize LLM
        vlm_config = self.config['vlm']
        self.llm = ChatOllama(
            model=vlm_config['model_name'],
            temperature=vlm_config['temperature']
        )
        
        # Video context
        self.current_video_data = None
        self.session_started = False
        
    def process_new_video(self, video_path: str) -> str:
        """Process a new video and prepare for conversation"""
        print(f"Processing new video: {video_path}")
        
        # Process video
        results = self.video_processor.process_video_complete(video_path)
        
        if results['processing_status'] != 'completed':
            return f"Error processing video: {results.get('error', 'Unknown error')}"
        
        # Load processed data
        self.current_video_data = self.video_processor.load_processed_data()
        
        # Add to vector store
        if self.current_video_data:
            # Add frame descriptions
            self.vector_store.add_frame_descriptions(
                self.current_video_data['important_frames']
            )
            
            # Add transcript segments
            self.vector_store.add_transcript_segments(
                self.current_video_data['transcript']['segments']
            )
        
        # Start new chat session with video context
        session_id = self.chat_memory.start_new_session()
        self.chat_memory.set_video_context(video_path, self.current_video_data)
        
        # Generate initial summary
        summary_response = self._generate_video_summary()
        
        # Add summary to chat history
        self.chat_memory.add_message(summary_response, 'assistant', {
            'type': 'video_summary',
            'video_path': video_path
        })
        
        self.session_started = True
        return summary_response
    
    def _generate_video_summary(self) -> str:
        """Generate automatic video summary using processed data"""
        if not self.current_video_data:
            return "No video data available for summary."
        
        metadata = self.current_video_data['video_metadata']
        transcript = self.current_video_data['transcript']
        important_frames = self.current_video_data['important_frames'][:5]  # Top 5 frames
        
        # Prepare context for LLM
        context = f"""
Video Duration: {int(metadata['duration'] // 60)}:{int(metadata['duration'] % 60):02d}
Total Frames Analyzed: {metadata['total_frames_analyzed']}
Important Moments Found: {metadata['important_frames_count']}

TRANSCRIPT:
{transcript['full_text'][:1000]}...

IMPORTANT VISUAL MOMENTS:
"""
        
        for frame in important_frames:
            timestamp = frame['timestamp']
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            context += f"\n• {minutes:02d}:{seconds:02d} - {frame['vlm_analysis'][:100]}..."
        
        summary_prompt = f"""Based on the video analysis below, provide a comprehensive summary of this video. Include:

1. **Video Overview**: What type of content is this and what's the main topic?
2. **Key Moments**: List the most important timestamps with brief descriptions
3. **Main Points**: What are the key takeaways or important information?
4. **Visual Elements**: What notable visual elements or scenes were observed?

Make the summary conversational and helpful for someone who wants to understand the video content quickly.

{context}

Provide a natural, engaging summary that I can use to start a conversation about this video."""
        
        try:
            response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            return response.content
        except Exception as e:
            return f"Unable to generate summary: {str(e)}. However, the video has been processed and is ready for questions."
    
    def chat(self, user_message: str) -> str:
        """Main chat interface"""
        if not self.session_started:
            return "Please process a video first using process_new_video(video_path) before starting a chat."
        
        # Add user message to memory
        self.chat_memory.add_message(user_message, 'user')
        
        # Get conversation context
        context = self.chat_memory.get_conversation_context()
        
        # Search for relevant content
        relevant_content = self._search_relevant_content(user_message)
        
        # Generate response
        response = self._generate_response(user_message, context, relevant_content)
        
        # Add response to memory
        self.chat_memory.add_message(response, 'assistant')
        
        # Save session
        self.chat_memory.save_current_session()
        
        return response
    
    def _search_relevant_content(self, query: str) -> Dict:
        """Search for relevant video content based on query"""
        if not self.current_video_data:
            return {'frames': [], 'transcript_segments': []}
        
        try:
            # Search in vector store
            search_results = self.vector_store.search_by_query(query, k=5)
            
            relevant_frames = []
            relevant_transcript = []
            
            for result in search_results:
                if result.metadata.get('type') == 'frame_analysis':
                    relevant_frames.append({
                        'timestamp': result.metadata.get('timestamp'),
                        'content': result.page_content,
                        'importance_score': result.metadata.get('importance_score')
                    })
                elif result.metadata.get('type') == 'transcript_segment':
                    relevant_transcript.append({
                        'start_time': result.metadata.get('start_time'),
                        'end_time': result.metadata.get('end_time'),
                        'text': result.page_content
                    })
            
            return {
                'frames': relevant_frames[:3],  # Top 3 relevant frames
                'transcript_segments': relevant_transcript[:3]  # Top 3 relevant segments
            }
            
        except Exception as e:
            print(f"Search error: {e}")
            return {'frames': [], 'transcript_segments': []}
    
    def _generate_response(self, user_message: str, context: Dict, relevant_content: Dict) -> str:
        """Generate contextual response using LLM"""
        
        # Build context prompt
        context_prompt = f"""You are a video analysis assistant. You have processed a video and can answer questions about it using the following context:

CURRENT VIDEO: {context.get('video_context', {}).get('video_path', 'Unknown')}
CONVERSATION HISTORY:
"""
        
        # Add recent messages
        for msg in context.get('messages', [])[-5:]:  # Last 5 messages
            role = msg['type'].upper()
            content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            context_prompt += f"{role}: {content}\n"
        
        # Add relevant content
        if relevant_content['frames']:
            context_prompt += "\nRELEVANT VISUAL MOMENTS:\n"
            for frame in relevant_content['frames']:
                timestamp = frame['timestamp']
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                context_prompt += f"• {minutes:02d}:{seconds:02d} - {frame['content'][:150]}...\n"
        
        if relevant_content['transcript_segments']:
            context_prompt += "\nRELEVANT TRANSCRIPT SEGMENTS:\n"
            for segment in relevant_content['transcript_segments']:
                start_time = segment['start_time']
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                context_prompt += f"• {minutes:02d}:{seconds:02d} - {segment['text'][:150]}...\n"
        
        context_prompt += f"""
USER QUESTION: {user_message}

Please provide a helpful, specific answer based on the video content. Include specific timestamps when relevant. Be conversational and engaging."""
        
        try:
            response = self.llm.invoke([HumanMessage(content=context_prompt)])
            return response.content
        except Exception as e:
            return f"I encountered an error generating a response: {str(e)}. Please try rephrasing your question."
    
    def get_timestamp_context(self, timestamp: float) -> str:
        """Get detailed context around a specific timestamp"""
        if not self.current_video_data:
            return "No video data loaded."
        
        context_data = self.video_processor.get_context_for_timestamp(timestamp, context_window=30)
        
        if 'error' in context_data:
            return context_data['error']
        
        response = f"Context around {int(timestamp//60):02d}:{int(timestamp%60):02d}:\n\n"
        
        for item in context_data['relevant_items']:
            if item['type'] == 'transcript':
                start_time = item['start_time']
                response += f"🎵 {int(start_time//60):02d}:{int(start_time%60):02d} - {item['content']}\n"
            elif item['type'] == 'important_frame':
                frame_time = item['timestamp']
                response += f"📸 {int(frame_time//60):02d}:{int(frame_time%60):02d} - {item['content'][:100]}...\n"
        
        return response
    
    def list_important_moments(self) -> str:
        """List all important moments in the video"""
        if not self.current_video_data:
            return "No video data loaded."
        
        important_frames = self.current_video_data['important_frames']
        
        response = f"Important moments in the video ({len(important_frames)} total):\n\n"
        
        for i, frame in enumerate(important_frames[:10], 1):  # Top 10
            timestamp = frame['timestamp']
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            score = frame['importance_score']
            description = frame['vlm_analysis'][:100] + "..." if len(frame['vlm_analysis']) > 100 else frame['vlm_analysis']
            
            response += f"{i}. {minutes:02d}:{seconds:02d} (score: {score:.2f})\n   {description}\n\n"
        
        return response


# Example usage and testing
if __name__ == "__main__":
    chat_bot = EnhancedVideoChat()
    
    # Process video
    video_path = "data/test_sample5.mp4"
    if os.path.exists(video_path):
        print("Processing video...")
        summary = chat_bot.process_new_video(video_path)
        print("="*50)
        print("VIDEO SUMMARY")
        print("="*50)
        print(summary)
        print("\n" + "="*50)
        print("Ready for chat! Try asking questions about the video.")
        print("="*50)
    else:
        print(f"Video file not found: {video_path}")