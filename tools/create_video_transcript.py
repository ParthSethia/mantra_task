import whisper

def get_transcript(audio_path, model_name = 'large', output_path = 'cache/transcript_2.txt'):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    
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