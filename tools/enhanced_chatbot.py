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
from utils.graph_store import VideoGraphStore

class EnhancedVideoChat:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.video_processor = VideoProcessor(config_path)
        self.chat_memory = ChatMemory(config_path)
        self.vector_store = VideoVectorStore(config_path)
        self.graph_store = VideoGraphStore(config_path)
        
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
        
        if results['processing_status'] not in ['completed', 'completed_from_cache']:
            return f"Error processing video: {results.get('error', 'Unknown error')}"
        
        # Check if loaded from cache
        if results['processing_status'] == 'completed_from_cache':
            cache_info = results.get('cache_info', {})
            print(f"✓ Loaded from cache (processed: {cache_info.get('processed_at', 'unknown')})")
        
        # Load processed data
        self.current_video_data = self.video_processor.load_processed_data(video_path)
        
        # Add to vector store
        if self.current_video_data:
            print(f"DEBUG: Current video data keys: {list(self.current_video_data.keys())}")
            
            # Add frame descriptions
            important_frames = self.current_video_data.get('important_frames', [])
            print(f"DEBUG: Adding {len(important_frames)} important frames to vector store")
            self.vector_store.add_frame_descriptions(important_frames)
            
            # Add transcript segments
            transcript_data = self.current_video_data.get('transcript', {})
            transcript_segments = transcript_data.get('segments', [])
            print(f"DEBUG: Found transcript data: {bool(transcript_data)}")
            print(f"DEBUG: Found {len(transcript_segments)} transcript segments")
            
            if transcript_segments:
                print(f"DEBUG: First transcript segment: {transcript_segments[0]}")
                self.vector_store.add_transcript_segments(transcript_segments)
            else:
                print("DEBUG: No transcript segments found to add to vector store")
            
            # Load knowledge graph
            graph_file = os.path.join(self.video_processor.cache_config['metadata_directory'], 'knowledge_graph.json')
            if os.path.exists(graph_file):
                self.graph_store.load_graph(graph_file)
        
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
        
        # Prepare context for LLM - adapt based on whether transcript exists
        has_transcript = transcript.get('has_transcript', bool(transcript.get('segments')))
        
        context = f"""
Video Duration: {int(metadata['duration'] // 60)}:{int(metadata['duration'] % 60):02d}
Total Frames Analyzed: {metadata['total_frames_analyzed']}
Important Moments Found: {metadata['important_frames_count']}
Content Type: {'Video with Audio/Speech' if has_transcript else 'Visual-Only Video (e.g., CCTV/Surveillance)'}
"""

        if has_transcript:
            context += f"""
TRANSCRIPT:
{transcript['full_text'][:1000]}...
"""

        context += "\nIMPORTANT VISUAL MOMENTS:"
        
        for frame in important_frames:
            timestamp = frame['timestamp']
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            context += f"\n• {minutes:02d}:{seconds:02d} - {frame['vlm_analysis'][:100]}..."
        
        # Adapt summary prompt based on content type
        if has_transcript:
            summary_prompt = f"""Based on the video analysis below, provide a comprehensive summary of this video. Include:

1. **Video Overview**: What type of content is this and what's the main topic?
2. **Key Moments**: List the most important timestamps with brief descriptions
3. **Main Points**: What are the key takeaways or important information from both audio and visual content?
4. **Visual Elements**: What notable visual elements or scenes were observed?

Make the summary conversational and helpful for someone who wants to understand the video content quickly.

{context}

Provide a natural, engaging summary that I can use to start a conversation about this video."""
        else:
            summary_prompt = f"""Based on the visual analysis below, provide a comprehensive summary of this video. This appears to be a visual-only video (like CCTV footage) without meaningful audio content. Include:

1. **Video Overview**: What type of visual content is this? (surveillance, security footage, etc.)
2. **Key Visual Moments**: List the most important timestamps with descriptions of what's happening visually
3. **Visual Activity**: What are the main activities, movements, or events observed?
4. **Scene Elements**: What objects, people, or notable visual elements appear throughout?

Focus entirely on visual content and make the summary helpful for someone monitoring or reviewing this footage.

{context}

Provide a clear, practical summary focused on visual events and activities."""
        
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
            search_results = self.vector_store.search_by_query(query, k=10)  # Increased k to get more results
            
            relevant_frames = []
            relevant_transcript = []
            
            print(f"DEBUG: Vector search returned {len(search_results)} results for query: '{query}'")
            
            for i, result in enumerate(search_results):
                result_type = result.metadata.get('type', 'unknown')
                print(f"DEBUG: Result {i}: type='{result_type}', content preview='{result.page_content[:100]}...'")
                
                if result_type == 'frame_analysis':
                    relevant_frames.append({
                        'timestamp': result.metadata.get('timestamp'),
                        'content': result.page_content,
                        'importance_score': result.metadata.get('importance_score')
                    })
                elif result_type == 'transcript_segment':
                    relevant_transcript.append({
                        'start_time': result.metadata.get('start_time'),
                        'end_time': result.metadata.get('end_time'),
                        'text': result.page_content
                    })
                else:
                    print(f"DEBUG: Unknown result type: {result_type}")
            
            print(f"DEBUG: Found {len(relevant_frames)} frame results, {len(relevant_transcript)} transcript results")
            
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
        has_transcript = len(relevant_content.get('transcript_segments', [])) > 0
        has_visual = len(relevant_content.get('frames', [])) > 0
        
        context_prompt = f"""You are a video analysis assistant. You have processed a video and can answer questions about it using BOTH visual analysis and transcript information.

CURRENT VIDEO: {context.get('video_context', {}).get('video_path', 'Unknown')}
AVAILABLE CONTENT: {"Transcript + Visual" if has_transcript and has_visual else "Transcript Only" if has_transcript else "Visual Only" if has_visual else "Limited"}

IMPORTANT: When answering questions, you should:
- Use BOTH transcript content and visual analysis when available
- For questions about "transcript information" or "what was said", prioritize transcript content
- For questions about "visual moments" or "what was seen", prioritize visual analysis  
- For general questions, combine both types of information for comprehensive answers
- Always mention timestamps when referencing specific moments

CONVERSATION HISTORY:
"""
        
        # Add recent messages
        for msg in context.get('messages', [])[-5:]:  # Last 5 messages
            role = msg['type'].upper()
            content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            context_prompt += f"{role}: {content}\n"
        
        # Add relevant content
        print(f"DEBUG: Building context with {len(relevant_content['frames'])} frames, {len(relevant_content['transcript_segments'])} transcript segments")
        
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
            print(f"DEBUG: Added {len(relevant_content['transcript_segments'])} transcript segments to context")
        else:
            print("DEBUG: No transcript segments found in relevant content")
        
        context_prompt += f"""
USER QUESTION: {user_message}

Please provide a helpful, specific answer based on the video content. Include specific timestamps when relevant. Be conversational and engaging."""
        
        try:
            # Check if this is a temporal query
            temporal_keywords = ['before', 'after', 'together', 'same time', 'co-occur']
            if any(keyword in user_message.lower() for keyword in temporal_keywords):
                temporal_response = self.handle_temporal_query(user_message)
                return temporal_response
            
            # Check if this is an object-related query
            object_keywords = ['object', 'track', 'car', 'person', 'people', 'vehicle']
            if any(keyword in user_message.lower() for keyword in object_keywords):
                object_info = self.get_object_information(user_message)
                if "No object tracking data" not in object_info:
                    return object_info
            
            response = self.llm.invoke([HumanMessage(content=context_prompt)])
            return response.content
        except Exception as e:
            return f"I encountered an error generating a response: {str(e)}. Please try rephrasing your question."
    
    def handle_temporal_query(self, user_message: str) -> str:
        """Handle temporal reasoning queries using graph database"""
        try:
            # Use graph store to process temporal queries
            query_result = self.graph_store.query_temporal_relationships(user_message)
            
            if query_result['query_type'] == 'before':
                events = query_result.get('events', [])
                if events:
                    response = f"Events before {query_result['timestamp']//60:.0f}:{query_result['timestamp']%60:02.0f}:\n\n"
                    for event in events:
                        timestamp = event['timestamp']
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        
                        if event['node_type'] == 'frame':
                            response += f"📸 {minutes:02d}:{seconds:02d} - {event['data']['analysis'][:100]}...\n"
                        elif event['node_type'] == 'transcript_segment':
                            response += f"🎵 {minutes:02d}:{seconds:02d} - {event['data']['text'][:100]}...\n"
                        elif event['node_type'] == 'object':
                            response += f"👀 {minutes:02d}:{seconds:02d} - {event['data']['class_name']} detected\n"
                    return response
                else:
                    return f"No events found before the specified time."
            
            elif query_result['query_type'] == 'after':
                events = query_result.get('events', [])
                if events:
                    response = f"Events after {query_result['timestamp']//60:.0f}:{query_result['timestamp']%60:02.0f}:\n\n"
                    for event in events:
                        timestamp = event['timestamp']
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        
                        if event['node_type'] == 'frame':
                            response += f"📸 {minutes:02d}:{seconds:02d} - {event['data']['analysis'][:100]}...\n"
                        elif event['node_type'] == 'transcript_segment':
                            response += f"🎵 {minutes:02d}:{seconds:02d} - {event['data']['text'][:100]}...\n"
                        elif event['node_type'] == 'object':
                            response += f"👀 {minutes:02d}:{seconds:02d} - {event['data']['class_name']} detected\n"
                    return response
                else:
                    return f"No events found after the specified time."
            
            elif query_result['query_type'] == 'co_occurring':
                objects = query_result.get('objects', [])
                if objects:
                    response = f"Objects present around {query_result['timestamp']//60:.0f}:{query_result['timestamp']%60:02.0f}:\n\n"
                    for obj in objects:
                        response += f"• {obj['class_name']} (Track {obj['track_id']}) - Active from "
                        response += f"{obj['first_seen']//60:.0f}:{obj['first_seen']%60:02.0f} to "
                        response += f"{obj['last_seen']//60:.0f}:{obj['last_seen']%60:02.0f}\n"
                    return response
                else:
                    return f"No objects found around the specified time."
            
            else:
                return query_result.get('message', 'Query not understood. Try asking about events before/after a timestamp.')
                
        except Exception as e:
            return f"Error processing temporal query: {str(e)}"
    
    def get_object_information(self, query: str) -> str:
        """Get information about tracked objects"""
        if not self.current_video_data or not self.current_video_data.get('object_tracking'):
            return "No object tracking data available for this video."
        
        tracking_data = self.current_video_data['object_tracking']
        summary = tracking_data.get('summary', {})
        
        query_lower = query.lower()
        
        if 'objects' in query_lower or 'tracks' in query_lower:
            response = f"Object Tracking Summary:\n\n"
            response += f"Total tracks found: {summary.get('total_tracks', 0)}\n"
            response += f"Active tracks: {summary.get('active_tracks', 0)}\n\n"
            
            track_classes = summary.get('track_classes', {})
            if track_classes:
                response += "Objects detected:\n"
                for class_name, count in track_classes.items():
                    response += f"• {class_name}: {count} tracks\n"
            
            return response
        
        # Look for specific object classes in the query
        if tracking_data.get('tracks'):
            relevant_tracks = []
            for track_id, track_data in tracking_data['tracks'].items():
                class_name = track_data.get('class_name', '')
                if class_name.lower() in query_lower or f"track {track_id}" in query_lower:
                    relevant_tracks.append(track_data)
            
            if relevant_tracks:
                response = f"Found {len(relevant_tracks)} relevant tracks:\n\n"
                for track in relevant_tracks[:5]:  # Limit to 5 tracks
                    first_seen = track.get('first_seen', 0)
                    last_seen = track.get('last_seen', 0)
                    response += f"• {track.get('class_name', 'Unknown')} (Track {track.get('track_id')})\n"
                    response += f"  Active: {first_seen//60:.0f}:{first_seen%60:02.0f} - {last_seen//60:.0f}:{last_seen%60:02.0f}\n"
                    response += f"  Detections: {len(track.get('detections', []))}\n\n"
                
                return response
            else:
                return f"No tracks found matching your query. Available objects: {', '.join(summary.get('track_classes', {}).keys())}"
        
        return "No object tracking data available."
    
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