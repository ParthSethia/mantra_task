import os
import yaml
import json
from typing import Dict, List, Optional

from load_video import get_audio_from_video
from create_video_transcript import get_transcript, get_video_id
from video_frame_processor import VideoFrameProcessor
from object_tracker import ObjectTracker
from utils.graph_store import VideoGraphStore
from utils.video_cache import VideoCacheManager

class VideoProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.video_config = self.config['video_processing']
        self.audio_config = self.config['audio_processing']
        self.cache_config = self.config['cache']
        
        # Initialize components
        self.frame_processor = VideoFrameProcessor(config_path)
        self.object_tracker = ObjectTracker(config_path)
        self.graph_store = VideoGraphStore(config_path)
        self.cache_manager = VideoCacheManager(config_path)
        
        # Create cache directory
        os.makedirs(self.cache_config['directory'], exist_ok=True)
    
    def process_video_complete(self, video_path: str) -> Dict:
        print(f"Starting complete video processing for: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check cache first
        cached_data = self.cache_manager.get_cached_data(video_path)
        if cached_data:
            print("✓ Using cached video processing results")
            return {
                'video_path': video_path,
                'processing_status': 'completed_from_cache',
                'components': {
                    'cache': {'status': 'loaded', 'source': 'cache'},
                    'audio': {'status': 'cached'},
                    'transcript': {'status': 'cached'},
                    'frames': {'status': 'cached'},
                    'tracking': {'status': 'cached'},
                    'graph': {'status': 'cached'}
                },
                'summary': self._generate_processing_summary(cached_data),
                'cache_info': cached_data.get('cache_info', {})
            }
        
        results = {
            'video_path': video_path,
            'processing_status': 'started',
            'components': {}
        }
        
        try:
            # Step 1: Extract audio
            print("Step 1: Extracting audio...")
            audio_path = self.audio_config['output_path']
            get_audio_from_video(video_path, audio_path)
            results['components']['audio'] = {
                'status': 'completed',
                'output_path': audio_path
            }
            print(f"Audio extracted to: {audio_path}")
            
            # Step 2: Create transcript (if audio processing enabled)
            transcript_result = {'segments': []}
            if self.audio_config.get('enabled', True):
                print("Step 2: Creating transcript...")
                
                # Check if audio actually contains speech
                if self._has_meaningful_audio(audio_path):
                    transcript_path = self.audio_config['transcript_output_path']
                    transcript_result = get_transcript(
                        audio_path, 
                        model_name=self.audio_config['whisper_model_size'],
                        output_path=transcript_path,
                        video_path=video_path  # Pass video path for video-specific caching
                    )
                    results['components']['transcript'] = {
                        'status': 'completed',
                        'output_path': transcript_path,
                        'segments_count': len(transcript_result['segments'])
                    }
                    print(f"Transcript created with {len(transcript_result['segments'])} segments")
                else:
                    print("No meaningful audio detected, skipping transcription")
                    results['components']['transcript'] = {
                        'status': 'skipped',
                        'reason': 'no_meaningful_audio'
                    }
            else:
                print("Step 2: Audio processing disabled, skipping transcript")
                results['components']['transcript'] = {
                    'status': 'disabled'
                }
            
            # Step 3: Process frames with VLM
            print("Step 3: Processing video frames...")
            analyzed_frames = self.frame_processor.process_video_frames(video_path)
            important_frames = self.frame_processor.get_important_frames(analyzed_frames)
            
            # Use video-specific analysis filename
            video_id = get_video_id(video_path)
            results['components']['frames'] = {
                'status': 'completed',
                'total_frames': len(analyzed_frames),
                'important_frames': len(important_frames),
                'analysis_file': os.path.join(self.cache_config['metadata_directory'], f'frame_analysis_{video_id}.json')
            }
            
            # Step 4: Object tracking (if enabled)
            tracking_data = None
            if self.object_tracker.tracking_config['enabled']:
                print("Step 4: Object tracking and detection...")
                tracking_data = self._process_object_tracking(video_path, video_id)
                
                results['components']['tracking'] = {
                    'status': 'completed' if tracking_data else 'failed',
                    'tracks_found': len(tracking_data.get('tracks', {})) if tracking_data else 0,
                    'tracking_file': os.path.join(self.cache_config['metadata_directory'], f'object_tracking_{video_id}.json')
                }
            else:
                print("Step 4: Object tracking disabled")
                results['components']['tracking'] = {'status': 'disabled'}
            
            # Step 5: Graph database population
            print("Step 5: Building knowledge graph...")
            self._build_knowledge_graph(transcript_result, analyzed_frames, tracking_data, video_id)
            
            results['components']['graph'] = {
                'status': 'completed',
                'graph_file': os.path.join(self.cache_config['metadata_directory'], f'knowledge_graph_{video_id}.json')
            }
            
            # Step 6: Create consolidated metadata
            print("Step 6: Creating consolidated metadata...")
            consolidated_data = self._create_consolidated_metadata(
                transcript_result, analyzed_frames, important_frames, tracking_data
            )
            
            # Use video-specific metadata filename
            video_id = get_video_id(video_path)
            metadata_file = os.path.join(self.cache_config['metadata_directory'], f'video_metadata_{video_id}.json')
            with open(metadata_file, 'w') as f:
                json.dump(consolidated_data, f, indent=2)
            
            results['components']['metadata'] = {
                'status': 'completed',
                'output_path': metadata_file
            }
            
            results['processing_status'] = 'completed'
            results['summary'] = self._generate_processing_summary(consolidated_data)
            
            # Cache the results for future use
            print("Step 7: Caching processed data...")
            self.cache_manager.cache_video_data(video_path, results, consolidated_data)
            
            print("Video processing completed successfully!")
            return results
            
        except Exception as e:
            print(f"Error during video processing: {e}")
            results['processing_status'] = 'failed'
            results['error'] = str(e)
            return results
    
    def _process_object_tracking(self, video_path: str, video_id: str) -> Optional[Dict]:
        """Process video for object tracking"""
        try:
            import cv2
            
            # Open video
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            tracking_results = []
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                timestamp = frame_count / fps
                
                # Process frame for tracking
                active_tracks, annotated_frame = self.object_tracker.process_frame(frame, timestamp)
                
                if active_tracks:
                    tracking_results.append({
                        'timestamp': timestamp,
                        'tracks': [
                            {
                                'track_id': track.track_id,
                                'class_name': track.class_name,
                                'bbox': track.detections[-1]['bbox'] if track.detections else None,
                                'confidence': track.detections[-1]['confidence'] if track.detections else 0.0
                            }
                            for track in active_tracks
                        ]
                    })
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames for tracking...")
            
            video.release()
            
            # Get tracking summary and save
            tracking_summary = self.object_tracker.get_track_summary()
            
            # Save tracking data with video-specific filename
            tracking_file = os.path.join(self.cache_config['metadata_directory'], f'object_tracking_{video_id}.json')
            self.object_tracker.save_tracking_data(tracking_file)
            
            print(f"Object tracking completed. Found {tracking_summary['total_tracks']} tracks")
            
            return {
                'summary': tracking_summary,
                'tracks': self.object_tracker.tracks,
                'frame_results': tracking_results
            }
            
        except Exception as e:
            print(f"Object tracking failed: {e}")
            return None
    
    def _build_knowledge_graph(self, transcript_result: Dict, analyzed_frames: List[Dict], 
                              tracking_data: Optional[Dict], video_id: str):
        """Build knowledge graph with all video data"""
        try:
            # Create video node
            video_metadata = {
                'duration': max(segment['end'] for segment in transcript_result['segments']) if transcript_result['segments'] else 0,
                'total_frames_analyzed': len(analyzed_frames),
                'important_frames_count': len([f for f in analyzed_frames if f.get('importance_score', 0) > 0.6]),
                'transcript_segments': len(transcript_result['segments'])
            }
            
            video_id = self.graph_store.create_video_node("current_video", video_metadata)
            
            # Add frame nodes
            frame_ids = self.graph_store.add_frame_nodes(analyzed_frames)
            
            # Add transcript nodes (only if transcript exists)
            transcript_ids = []
            if transcript_result['segments']:
                transcript_ids = self.graph_store.add_transcript_nodes(transcript_result['segments'])
                
                # Align frames with transcript
                self.graph_store.align_frames_with_transcript(frame_ids, transcript_ids)
            else:
                print("No transcript available - creating vision-only knowledge graph")
            
            # Add object tracking data if available
            if tracking_data and tracking_data.get('tracks'):
                self.graph_store.add_object_tracks(tracking_data)
            
            # Save graph with video-specific filename
            graph_file = os.path.join(self.cache_config['metadata_directory'], f'knowledge_graph_{video_id}.json')
            self.graph_store.save_graph(graph_file)
            
            # Print statistics
            stats = self.graph_store.get_graph_statistics()
            print(f"Knowledge graph built: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
            
        except Exception as e:
            print(f"Graph building failed: {e}")
    
    def _has_meaningful_audio(self, audio_path: str) -> bool:
        """Check if audio file contains meaningful content (not just silence)"""
        try:
            import librosa
            import numpy as np
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Check duration
            duration = len(audio) / sr
            min_duration = self.audio_config.get('min_audio_duration', 1.0)
            
            if duration < min_duration:
                print(f"Audio too short ({duration:.1f}s < {min_duration}s)")
                return False
            
            # Check if audio is mostly silence
            rms = librosa.feature.rms(y=audio)[0]
            mean_rms = np.mean(rms)
            
            # Convert to dB
            if mean_rms > 0:
                db_level = 20 * np.log10(mean_rms)
                threshold = self.audio_config.get('silence_threshold', -40)
                
                if db_level < threshold:
                    print(f"Audio mostly silent ({db_level:.1f}dB < {threshold}dB)")
                    return False
            
            print(f"Audio contains meaningful content ({duration:.1f}s, {db_level:.1f}dB)")
            return True
            
        except ImportError:
            print("librosa not available, assuming audio is meaningful")
            return True
        except Exception as e:
            print(f"Audio analysis failed: {e}, assuming audio is meaningful")
            return True
    
    def _create_consolidated_metadata(self, transcript_result: Dict, analyzed_frames: List[Dict], 
                                    important_frames: List[Dict], tracking_data: Optional[Dict]) -> Dict:
        # Get video duration - try transcript first, then estimate from frames
        duration = 0
        if transcript_result['segments']:
            duration = max(segment['end'] for segment in transcript_result['segments'])
        elif analyzed_frames:
            # Estimate duration from last frame timestamp
            duration = max(frame['timestamp'] for frame in analyzed_frames)
        
        # Create timeline with both transcript and important frames
        timeline = []
        
        # Add transcript segments (only if they exist)
        if transcript_result['segments']:
            for segment in transcript_result['segments']:
                timeline.append({
                    'type': 'transcript',
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'content': segment['text'].strip(),
                    'data': segment
                })
        
        # Add important frames
        for frame in important_frames:
            timeline.append({
                'type': 'important_frame',
                'timestamp': frame['timestamp'],
                'importance_score': frame['importance_score'],
                'content': frame['vlm_analysis'],
                'frame_path': frame['frame_path'],
                'data': frame
            })
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x.get('timestamp', x.get('start_time', 0)))
        
        # Add tracking summary if available
        tracking_summary = {}
        if tracking_data:
            tracking_summary = {
                'total_tracks': tracking_data.get('summary', {}).get('total_tracks', 0),
                'active_tracks': tracking_data.get('summary', {}).get('active_tracks', 0),
                'track_classes': dict(tracking_data.get('summary', {}).get('track_classes', {}))
            }
        
        return {
            'video_metadata': {
                'duration': duration,
                'total_frames_analyzed': len(analyzed_frames),
                'important_frames_count': len(important_frames),
                'transcript_segments': len(transcript_result['segments']),
                'tracking_summary': tracking_summary
            },
            'timeline': timeline,
            'transcript': {
                'full_text': ' '.join(segment['text'].strip() for segment in transcript_result['segments']) if transcript_result['segments'] else '',
                'segments': transcript_result['segments'],
                'has_transcript': bool(transcript_result['segments'])
            },
            'important_frames': important_frames,
            'object_tracking': tracking_data if tracking_data else {},
            'processing_config': {
                'frame_fps': self.video_config['frame_extraction_fps'],
                'importance_threshold': self.frame_processor.vlm_config['importance_threshold'],
                'whisper_model': self.audio_config['whisper_model_size'],
                'object_tracking_enabled': self.object_tracker.tracking_config['enabled']
            }
        }
    
    def _generate_processing_summary(self, consolidated_data: Dict) -> Dict:
        metadata = consolidated_data['video_metadata']
        
        # Find top moments
        important_moments = []
        for item in consolidated_data['timeline']:
            if item['type'] == 'important_frame' and item['importance_score'] > 0.7:
                important_moments.append({
                    'timestamp': item['timestamp'],
                    'score': item['importance_score'],
                    'description': item['content'][:100] + "..." if len(item['content']) > 100 else item['content']
                })
        
        important_moments.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'duration': f"{int(metadata['duration'] // 60)}:{int(metadata['duration'] % 60):02d}",
            'total_frames_processed': metadata['total_frames_analyzed'],
            'important_frames_found': metadata['important_frames_count'],
            'transcript_segments': metadata['transcript_segments'],
            'top_moments': important_moments[:5],  # Top 5 moments
            'processing_complete': True
        }
    
    def load_processed_data(self, video_path: str = None) -> Optional[Dict]:
        """Load previously processed video data"""
        # Try cache first if video_path provided
        if video_path:
            cached_data = self.cache_manager.get_cached_data(video_path)
            if cached_data:
                return cached_data
        
        # Fallback to default metadata file
        metadata_file = os.path.join(self.cache_config['metadata_directory'], 'video_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_context_for_timestamp(self, timestamp: float, context_window: int = 30) -> Dict:
        """Get relevant context (transcript + frames) around a specific timestamp"""
        consolidated_data = self.load_processed_data()
        if not consolidated_data:
            return {'error': 'No processed data found'}
        
        relevant_items = []
        
        for item in consolidated_data['timeline']:
            item_time = item.get('timestamp', item.get('start_time', 0))
            
            if abs(item_time - timestamp) <= context_window:
                relevant_items.append(item)
        
        relevant_items.sort(key=lambda x: x.get('timestamp', x.get('start_time', 0)))
        
        return {
            'target_timestamp': timestamp,
            'context_window': context_window,
            'relevant_items': relevant_items
        }


if __name__ == "__main__":
    processor = VideoProcessor()
    
    # Process the default video
    video_path = processor.video_config['input_path']
    
    if os.path.exists(video_path):
        print(f"Processing video: {video_path}")
        results = processor.process_video_complete(video_path)
        
        if results['processing_status'] == 'completed':
            print("\n" + "="*50)
            print("PROCESSING SUMMARY")
            print("="*50)
            summary = results['summary']
            print(f"Duration: {summary['duration']}")
            print(f"Frames processed: {summary['total_frames_processed']}")
            print(f"Important frames: {summary['important_frames_found']}")
            print(f"Transcript segments: {summary['transcript_segments']}")
            
            if summary['top_moments']:
                print("\nTop Moments:")
                for i, moment in enumerate(summary['top_moments'], 1):
                    print(f"{i}. {moment['timestamp']:.1f}s (score: {moment['score']:.2f})")
                    print(f"   {moment['description']}")
        else:
            print(f"Processing failed: {results.get('error', 'Unknown error')}")
    else:
        print(f"Video file not found: {video_path}")
        print("Please ensure the video file exists at the configured path.")