import os
import yaml
import json
from typing import Dict, List, Optional

from load_video import get_audio_from_video
from create_video_transcript import get_transcript
from video_frame_processor import VideoFrameProcessor

class VideoProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.video_config = self.config['video_processing']
        self.audio_config = self.config['audio_processing']
        self.cache_config = self.config['cache']
        
        # Initialize components
        self.frame_processor = VideoFrameProcessor(config_path)
        
        # Create cache directory
        os.makedirs(self.cache_config['directory'], exist_ok=True)
    
    def process_video_complete(self, video_path: str) -> Dict:
        print(f"Starting complete video processing for: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
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
            
            # Step 2: Create transcript
            print("Step 2: Creating transcript...")
            transcript_path = self.audio_config['transcript_output_path']
            transcript_result = get_transcript(
                audio_path, 
                model_name=self.audio_config['whisper_model_size'],
                output_path=transcript_path
            )
            results['components']['transcript'] = {
                'status': 'completed',
                'output_path': transcript_path,
                'segments_count': len(transcript_result['segments'])
            }
            print(f"Transcript created with {len(transcript_result['segments'])} segments")
            
            # Step 3: Process frames with VLM
            print("Step 3: Processing video frames...")
            analyzed_frames = self.frame_processor.process_video_frames(video_path)
            important_frames = self.frame_processor.get_important_frames(analyzed_frames)
            
            results['components']['frames'] = {
                'status': 'completed',
                'total_frames': len(analyzed_frames),
                'important_frames': len(important_frames),
                'analysis_file': os.path.join(self.cache_config['metadata_directory'], 'frame_analysis.json')
            }
            
            # Step 4: Create consolidated metadata
            print("Step 4: Creating consolidated metadata...")
            consolidated_data = self._create_consolidated_metadata(
                transcript_result, analyzed_frames, important_frames
            )
            
            metadata_file = os.path.join(self.cache_config['metadata_directory'], 'video_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(consolidated_data, f, indent=2)
            
            results['components']['metadata'] = {
                'status': 'completed',
                'output_path': metadata_file
            }
            
            results['processing_status'] = 'completed'
            results['summary'] = self._generate_processing_summary(consolidated_data)
            
            print("Video processing completed successfully!")
            return results
            
        except Exception as e:
            print(f"Error during video processing: {e}")
            results['processing_status'] = 'failed'
            results['error'] = str(e)
            return results
    
    def _create_consolidated_metadata(self, transcript_result: Dict, analyzed_frames: List[Dict], important_frames: List[Dict]) -> Dict:
        # Get video duration from transcript
        duration = max(segment['end'] for segment in transcript_result['segments']) if transcript_result['segments'] else 0
        
        # Create timeline with both transcript and important frames
        timeline = []
        
        # Add transcript segments
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
        
        return {
            'video_metadata': {
                'duration': duration,
                'total_frames_analyzed': len(analyzed_frames),
                'important_frames_count': len(important_frames),
                'transcript_segments': len(transcript_result['segments'])
            },
            'timeline': timeline,
            'transcript': {
                'full_text': ' '.join(segment['text'].strip() for segment in transcript_result['segments']),
                'segments': transcript_result['segments']
            },
            'important_frames': important_frames,
            'processing_config': {
                'frame_fps': self.video_config['frame_extraction_fps'],
                'importance_threshold': self.frame_processor.vlm_config['importance_threshold'],
                'whisper_model': self.audio_config['whisper_model_size']
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
    
    def load_processed_data(self) -> Optional[Dict]:
        """Load previously processed video data"""
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