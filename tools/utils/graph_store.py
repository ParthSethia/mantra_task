import networkx as nx
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import os

class VideoGraphStore:
    """Graph database for video temporal relationships and queries"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.graph_config = config['graph_database']
        self.cache_config = config['cache']
        
        # Initialize graph
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self.video_id = None
        
        # Node type definitions
        self.NODE_TYPES = {
            'VIDEO': 'video',
            'FRAME': 'frame', 
            'TRANSCRIPT': 'transcript_segment',
            'OBJECT': 'object',
            'TRACK': 'object_track',
            'SESSION': 'chat_session',
            'MESSAGE': 'chat_message'
        }
        
        # Relationship types
        self.RELATIONSHIP_TYPES = {
            'TEMPORAL_NEXT': 'temporal_next',     # Frame A follows Frame B
            'CONTAINS': 'contains',               # Video contains Frame/Transcript
            'ALIGNS_WITH': 'aligns_with',         # Frame aligns with Transcript
            'TRACKS': 'tracks',                   # Track contains multiple object instances
            'APPEARS_IN': 'appears_in',           # Object appears in Frame
            'REFERENCES': 'references',           # Message references Frame/Object/Timestamp
            'DISCUSSES': 'discusses',             # Session discusses Video moment
            'FOLLOWS': 'follows',                 # Sequential relationship
            'CO_OCCURS': 'co_occurs'              # Objects/events happen simultaneously
        }
        
        print(f"Graph database initialized using {self.graph_config['type']}")
    
    def create_video_node(self, video_path: str, metadata: Dict) -> str:
        """Create main video node"""
        video_id = f"video_{hash(video_path) % 10000}"
        self.video_id = video_id
        
        # Prepare node attributes avoiding conflicts
        node_attrs = {
            'node_type': self.NODE_TYPES['VIDEO'],
            'path': video_path,
            'video_duration': metadata.get('duration', 0),
            'total_frames': metadata.get('total_frames_analyzed', 0),
            'important_frames': metadata.get('important_frames_count', 0),
            'transcript_segments': metadata.get('transcript_segments', 0),
            'created_at': datetime.now().isoformat()
        }
        
        # Add other metadata without conflicts
        for key, value in metadata.items():
            if key not in node_attrs:
                node_attrs[key] = value
        
        self.graph.add_node(video_id, **node_attrs)
        
        return video_id
    
    def add_frame_nodes(self, frames_data: List[Dict]) -> List[str]:
        """Add frame nodes and temporal relationships"""
        frame_ids = []
        
        for i, frame in enumerate(frames_data):
            frame_id = f"frame_{frame['frame_id']}_{frame['timestamp']:.2f}s"
            
            self.graph.add_node(
                frame_id,
                node_type=self.NODE_TYPES['FRAME'],
                frame_id=frame['frame_id'],
                timestamp=frame['timestamp'],
                importance_score=frame['importance_score'],
                analysis=frame['vlm_analysis'],
                frame_path=frame.get('frame_path', ''),
                analyzed=frame.get('analyzed', False)
            )
            
            # Connect to video
            if self.video_id:
                self.graph.add_edge(
                    self.video_id, frame_id,
                    relationship=self.RELATIONSHIP_TYPES['CONTAINS']
                )
            
            # Add temporal relationships between consecutive frames
            if i > 0:
                prev_frame_id = frame_ids[i-1]
                self.graph.add_edge(
                    prev_frame_id, frame_id,
                    relationship=self.RELATIONSHIP_TYPES['TEMPORAL_NEXT'],
                    time_gap=frame['timestamp'] - frames_data[i-1]['timestamp']
                )
            
            frame_ids.append(frame_id)
        
        return frame_ids
    
    def add_transcript_nodes(self, transcript_segments: List[Dict]) -> List[str]:
        """Add transcript segment nodes"""
        transcript_ids = []
        
        for i, segment in enumerate(transcript_segments):
            transcript_id = f"transcript_{segment['start']:.2f}_{segment['end']:.2f}"
            
            self.graph.add_node(
                transcript_id,
                node_type=self.NODE_TYPES['TRANSCRIPT'],
                start_time=segment['start'],
                end_time=segment['end'],
                text=segment['text'],
                segment_duration=segment['end'] - segment['start']
            )
            
            # Connect to video
            if self.video_id:
                self.graph.add_edge(
                    self.video_id, transcript_id,
                    relationship=self.RELATIONSHIP_TYPES['CONTAINS']
                )
            
            # Add temporal relationships
            if i > 0:
                prev_transcript_id = transcript_ids[i-1]
                self.graph.add_edge(
                    prev_transcript_id, transcript_id,
                    relationship=self.RELATIONSHIP_TYPES['FOLLOWS']
                )
            
            transcript_ids.append(transcript_id)
        
        return transcript_ids
    
    def align_frames_with_transcript(self, frame_ids: List[str], transcript_ids: List[str]):
        """Create alignment relationships between frames and transcript segments"""
        for frame_id in frame_ids:
            frame_data = self.graph.nodes[frame_id]
            frame_timestamp = frame_data['timestamp']
            
            # Find overlapping transcript segments
            for transcript_id in transcript_ids:
                transcript_data = self.graph.nodes[transcript_id]
                start_time = transcript_data['start_time']
                end_time = transcript_data['end_time']
                
                # Check if frame timestamp falls within transcript segment
                if start_time <= frame_timestamp <= end_time:
                    self.graph.add_edge(
                        frame_id, transcript_id,
                        relationship=self.RELATIONSHIP_TYPES['ALIGNS_WITH'],
                        alignment_strength=1.0 - abs(frame_timestamp - (start_time + end_time) / 2) / (end_time - start_time)
                    )
    
    def add_object_tracks(self, tracking_data: Dict):
        """Add object tracking nodes and relationships"""
        if not tracking_data or 'tracks' not in tracking_data:
            return
        
        for track_id_str, track_data in tracking_data['tracks'].items():
            track_node_id = f"track_{track_data['track_id']}_{track_data['class_name']}"
            
            # Create track node
            self.graph.add_node(
                track_node_id,
                node_type=self.NODE_TYPES['TRACK'],
                track_id=track_data['track_id'],
                class_name=track_data['class_name'],
                first_seen=track_data['first_seen'],
                last_seen=track_data['last_seen'],
                track_duration=track_data['last_seen'] - track_data['first_seen'],
                detection_count=len(track_data['detections']),
                color=track_data['color'],
                active=track_data['active']
            )
            
            # Connect to video
            if self.video_id:
                self.graph.add_edge(
                    self.video_id, track_node_id,
                    relationship=self.RELATIONSHIP_TYPES['CONTAINS']
                )
            
            # Add object detection instances
            for i, detection in enumerate(track_data['detections']):
                object_instance_id = f"obj_{track_data['track_id']}_{detection['timestamp']:.2f}"
                
                self.graph.add_node(
                    object_instance_id,
                    node_type=self.NODE_TYPES['OBJECT'],
                    track_id=track_data['track_id'],
                    class_name=detection['class_name'],
                    timestamp=detection['timestamp'],
                    bbox=detection['bbox'],
                    confidence=detection['confidence']
                )
                
                # Connect instance to track
                self.graph.add_edge(
                    track_node_id, object_instance_id,
                    relationship=self.RELATIONSHIP_TYPES['TRACKS']
                )
                
                # Find corresponding frames and connect
                frame_nodes = [n for n in self.graph.nodes() 
                             if self.graph.nodes[n].get('node_type') == self.NODE_TYPES['FRAME']]
                
                for frame_id in frame_nodes:
                    frame_timestamp = self.graph.nodes[frame_id]['timestamp']
                    if abs(frame_timestamp - detection['timestamp']) < 0.5:  # Within 0.5 seconds
                        self.graph.add_edge(
                            object_instance_id, frame_id,
                            relationship=self.RELATIONSHIP_TYPES['APPEARS_IN'],
                            temporal_distance=abs(frame_timestamp - detection['timestamp'])
                        )
    
    def add_chat_session(self, session_data: Dict) -> str:
        """Add chat session and messages to graph"""
        session_id = f"session_{session_data['session_id']}"
        
        self.graph.add_node(
            session_id,
            node_type=self.NODE_TYPES['SESSION'],
            session_id=session_data['session_id'],
            start_time=session_data.get('start_time'),
            video_context=session_data.get('video_context', {}).get('video_path'),
            message_count=len(session_data.get('messages', []))
        )
        
        # Connect session to video if there's context
        if self.video_id and session_data.get('video_context'):
            self.graph.add_edge(
                session_id, self.video_id,
                relationship=self.RELATIONSHIP_TYPES['DISCUSSES']
            )
        
        # Add messages
        for i, message in enumerate(session_data.get('messages', [])):
            message_id = f"msg_{session_data['session_id']}_{i}"
            
            self.graph.add_node(
                message_id,
                node_type=self.NODE_TYPES['MESSAGE'],
                content=message['content'],
                type=message['type'],
                timestamp=message['timestamp']
            )
            
            # Connect message to session
            self.graph.add_edge(
                session_id, message_id,
                relationship=self.RELATIONSHIP_TYPES['CONTAINS']
            )
        
        return session_id
    
    def find_temporal_sequence(self, start_timestamp: float, end_timestamp: float) -> List[Dict]:
        """Find all events (frames, transcript, objects) in a time range"""
        events = []
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            node_type = node_data.get('node_type')
            
            # Check different timestamp fields based on node type
            timestamp = None
            if node_type == self.NODE_TYPES['FRAME']:
                timestamp = node_data.get('timestamp')
            elif node_type == self.NODE_TYPES['TRANSCRIPT']:
                # Use middle of transcript segment
                timestamp = (node_data.get('start_time', 0) + node_data.get('end_time', 0)) / 2
            elif node_type == self.NODE_TYPES['OBJECT']:
                timestamp = node_data.get('timestamp')
            
            if timestamp and start_timestamp <= timestamp <= end_timestamp:
                events.append({
                    'node_id': node_id,
                    'node_type': node_type,
                    'timestamp': timestamp,
                    'data': node_data
                })
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])
        return events
    
    def find_objects_in_timerange(self, start_time: float, end_time: float, 
                                  object_class: str = None) -> List[Dict]:
        """Find objects active in a specific time range"""
        objects = []
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            
            if node_data.get('node_type') == self.NODE_TYPES['TRACK']:
                first_seen = node_data.get('first_seen', 0)
                last_seen = node_data.get('last_seen', 0)
                
                # Check if track overlaps with time range
                if not (last_seen < start_time or first_seen > end_time):
                    if not object_class or node_data.get('class_name') == object_class:
                        objects.append({
                            'track_id': node_data.get('track_id'),
                            'class_name': node_data.get('class_name'),
                            'first_seen': first_seen,
                            'last_seen': last_seen,
                            'overlap_start': max(start_time, first_seen),
                            'overlap_end': min(end_time, last_seen),
                            'node_data': node_data
                        })
        
        return objects
    
    def find_events_before_timestamp(self, timestamp: float, limit: int = 5) -> List[Dict]:
        """Find events that happened before a specific timestamp"""
        return self.find_temporal_sequence(0, timestamp)[-limit:]
    
    def find_events_after_timestamp(self, timestamp: float, limit: int = 5) -> List[Dict]:
        """Find events that happened after a specific timestamp"""
        all_events = self.find_temporal_sequence(timestamp, float('inf'))
        return all_events[:limit]
    
    def find_co_occurring_objects(self, timestamp: float, time_window: float = 2.0) -> List[Dict]:
        """Find objects that appear together around a timestamp"""
        start_time = timestamp - time_window / 2
        end_time = timestamp + time_window / 2
        
        return self.find_objects_in_timerange(start_time, end_time)
    
    def query_temporal_relationships(self, query: str) -> Dict:
        """Process natural language temporal queries"""
        query_lower = query.lower()
        
        # Simple pattern matching for common queries
        if "before" in query_lower:
            # Extract timestamp if mentioned
            import re
            time_matches = re.findall(r'(\d+):(\d+)', query)
            if time_matches:
                minutes, seconds = map(int, time_matches[0])
                timestamp = minutes * 60 + seconds
                events = self.find_events_before_timestamp(timestamp)
                return {
                    'query_type': 'before',
                    'timestamp': timestamp,
                    'events': events
                }
        
        elif "after" in query_lower:
            time_matches = re.findall(r'(\d+):(\d+)', query)
            if time_matches:
                minutes, seconds = map(int, time_matches[0])
                timestamp = minutes * 60 + seconds
                events = self.find_events_after_timestamp(timestamp)
                return {
                    'query_type': 'after',
                    'timestamp': timestamp,
                    'events': events
                }
        
        elif "together" in query_lower or "same time" in query_lower:
            time_matches = re.findall(r'(\d+):(\d+)', query)
            if time_matches:
                minutes, seconds = map(int, time_matches[0])
                timestamp = minutes * 60 + seconds
                objects = self.find_co_occurring_objects(timestamp)
                return {
                    'query_type': 'co_occurring',
                    'timestamp': timestamp,
                    'objects': objects
                }
        
        # Default: return summary
        return {
            'query_type': 'unknown',
            'message': 'Query not recognized. Try phrases like "what happened before 2:30" or "objects at same time as 1:15"'
        }
    
    def get_graph_statistics(self) -> Dict:
        """Get graph statistics and metrics"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': defaultdict(int),
            'relationship_types': defaultdict(int),
            'temporal_span': {}
        }
        
        # Count node types
        for node_id in self.graph.nodes():
            node_type = self.graph.nodes[node_id].get('node_type', 'unknown')
            stats['node_types'][node_type] += 1
        
        # Count relationship types
        for edge in self.graph.edges(data=True):
            rel_type = edge[2].get('relationship', 'unknown')
            stats['relationship_types'][rel_type] += 1
        
        # Calculate temporal span
        timestamps = []
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if 'timestamp' in node_data:
                timestamps.append(node_data['timestamp'])
            elif 'start_time' in node_data:
                timestamps.append(node_data['start_time'])
        
        if timestamps:
            stats['temporal_span'] = {
                'start': min(timestamps),
                'end': max(timestamps),
                'span_duration': max(timestamps) - min(timestamps)
            }
        
        return stats
    
    def save_graph(self, filepath: str):
        """Save graph to file"""
        # Convert to JSON-serializable format
        graph_data = {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True)),
            'video_id': self.video_id,
            'config': self.graph_config
        }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
        
        print(f"Graph saved to: {filepath}")
    
    def load_graph(self, filepath: str) -> bool:
        """Load graph from file"""
        try:
            with open(filepath, 'r') as f:
                graph_data = json.load(f)
            
            # Reconstruct graph
            self.graph = nx.MultiDiGraph()
            
            # Add nodes
            for node_id, node_attrs in graph_data['nodes'].items():
                self.graph.add_node(node_id, **node_attrs)
            
            # Add edges
            for edge in graph_data['edges']:
                source, target, attrs = edge
                self.graph.add_edge(source, target, **attrs)
            
            self.video_id = graph_data.get('video_id')
            
            print(f"Graph loaded from: {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to load graph: {e}")
            return False


if __name__ == "__main__":
    # Test graph store
    graph_store = VideoGraphStore()
    
    print("Graph store initialized")
    print("Available query types:")
    print("- 'What happened before 2:30?'")
    print("- 'What occurred after 1:15?'") 
    print("- 'What objects were together at 3:45?'")
    print("Ready for video graph analysis...")