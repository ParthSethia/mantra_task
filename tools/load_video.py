from moviepy import VideoFileClip

def get_audio_from_video(video_path, audio_path = 'cache/audio.mp3'):
    """
    Loads and plays a video file with its associated audio.

    Args:
        video_path (str): The path to the video file.
    """
    try:
        video_clip = VideoFileClip(video_path)
        # video_clip.preview()  # Plays the video with audio
        # video_clip.close()
        audio = video_clip.audio
        audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"Error playing video: {e}")

# Example usage:
# play_video_with_audio("my_video.mp4")