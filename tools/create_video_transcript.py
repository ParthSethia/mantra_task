import whisper
import os
import re
import hashlib

def get_video_id(video_path):
    """Generate a unique ID for the video - consistent with cache manager"""
    # Use absolute path to ensure uniqueness (same as VideoCacheManager)
    abs_path = os.path.abspath(video_path)
    return hashlib.md5(abs_path.encode()).hexdigest()[:16]

def parse_existing_transcript(transcript_path):
    """Parse existing transcript file and convert to whisper-like format"""
    if not os.path.exists(transcript_path):
        return None
    
    segments = []
    segment_id = 0
    
    print(f"Loading existing transcript from: {transcript_path}")
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse format: [MM:SS - MM:SS] Text
            match = re.match(r'\[(\d{2}):(\d{2}) - (\d{2}):(\d{2})\]\s*(.*)', line)
            if match:
                start_min, start_sec, end_min, end_sec, text = match.groups()
                
                start_time = int(start_min) * 60 + int(start_sec)
                end_time = int(end_min) * 60 + int(end_sec)
                
                segments.append({
                    "id": segment_id,
                    "seek": start_time * 100,  # Whisper format
                    "start": float(start_time),
                    "end": float(end_time),
                    "text": text.strip(),
                    "tokens": [],  # Empty for manually created transcript
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0
                })
                segment_id += 1
    
    # Create whisper-like result object
    result = {
        "text": " ".join([seg["text"] for seg in segments]),
        "segments": segments,
        "language": "hi"  # Assuming Hindi based on your transcript
    }
    
    print(f"Loaded {len(segments)} segments from existing transcript")
    return result

def get_transcript(audio_path, model_name='large', output_path=None, video_path=None):
    # Generate video-specific transcript paths
    if video_path:
        video_id = get_video_id(video_path)
        existing_transcript_path = f'cache/transcript_{video_id}.txt'
        if output_path is None:
            output_path = f'cache/transcript_{video_id}_generated.txt'
        print(f"Video ID: {video_id}")
    else:
        # Fallback to generic names if video_path not provided
        existing_transcript_path = 'cache/transcript.txt'
        if output_path is None:
            output_path = 'cache/transcript_2.txt'
    
    # Check if existing video-specific transcript exists
    if os.path.exists(existing_transcript_path):
        print(f"Found existing transcript at {existing_transcript_path}, using it instead of creating new one")
        return parse_existing_transcript(existing_transcript_path)
    
    # Fallback to creating new transcript with whisper
    print(f"No existing transcript found at {existing_transcript_path}, creating new one with Whisper")
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            
            start_minutes = int(start_time // 60)
            start_seconds = int(start_time % 60)
            end_minutes = int(end_time // 60)
            end_seconds = int(end_time % 60)
            
            timestamp = f"[{start_minutes:02d}:{start_seconds:02d} - {end_minutes:02d}:{end_seconds:02d}]"
            f.write(f"{timestamp} {text}\n")
    
    print(f"Transcript with timestamps saved to {output_path}")
    return result