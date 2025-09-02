# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Multimodal Chat Assistant** project designed to process video input, recognize events, and provide intelligent summarization with multi-turn conversation capabilities. The system supports visual understanding across various domains including traffic monitoring, podcast analysis, educational content, and general video summarization to identify key moments and important segments.

## Project Structure

```
Mantra Softech/
├── Assignment_ Multimodal Chat Assistant.pdf  # Project requirements and specifications
└── tools/                                     # Video processing utilities
    ├── create_video_transcript.py             # Video transcription functionality (empty)
    └── load_video.py                          # Video loading utilities (empty)
```

## Key Requirements

- **Video Processing**: Handle 10-minute videos at 60 fps with efficient frame processing
- **Event Recognition**: Identify key moments and specific events within video streams
- **Intelligent Summarization**: Extract important segments from various video types (podcasts, educational content, surveillance footage)
- **Guideline Adherence**: Detect and report violations or adherence to predefined rules when applicable
- **Multi-turn Conversations**: Maintain context across conversation turns
- **Performance**: Target <2 seconds latency for query processing
- **VLM Integration**: Prioritize open-source Vision-Language Models supporting >512k tokens

## Development Notes

- This appears to be an early-stage project with placeholder Python files in the tools/ directory
- No package management files (requirements.txt, package.json) are present yet
- No existing build, test, or development commands are configured
- Architecture needs to be designed to support real-time video processing and VLM integration

## Test Data

Test video samples are available at: https://drive.google.com/drive/folders/1RYil2fWSlkKsmf65CwSav3H4h5DDHikl

## Implementation Focus Areas

1. **Video Stream Processing**: Efficient handling of high-throughput video data
2. **Content Analysis**: Computer vision and audio processing for identifying key moments, events, and important segments
3. **Multi-Domain Summarization**: Adaptive summarization for different content types (traffic scenes, podcasts, educational videos, etc.)
4. **Conversational AI**: Multi-turn chat with context retention and follow-up question handling
5. **Performance Optimization**: Real-time processing with low latency
6. **Guideline Engine**: Configurable rule-based evaluation when applicable to content type