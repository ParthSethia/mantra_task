# Multimodal Video Chat Assistant - Implementation Plan

## Project Overview
Building a comprehensive video chat assistant that processes video content, identifies key moments, tracks objects with persistent IDs, and provides intelligent summarization with multi-turn conversational capabilities.

## Current Codebase Status

### Existing Components ✅
- **Video Processing**: `tools/load_video.py` - Basic audio extraction from video
- **Transcription**: `tools/create_video_transcript.py` - Whisper-based transcript with timestamps
- **Vector Store**: `tools/utils/vector_store.py` - Milvus integration ready
- **Embeddings**: `tools/utils/embedding.py` - Ollama embeddings setup
- **Chat Memory**: `tools/utils/chat_memory.py` - LangGraph persistence URL (placeholder)
- **Chatbot**: `tools/chatbot.py` - LangChain + Ollama multimodal setup
- **Configuration**: `config.yaml` - Basic video/audio paths

### Project Structure
```
Mantra Softech/
├── Assignment_ Multimodal Chat Assistant.pdf
├── CLAUDE.md
├── cache/
│   ├── audio.mp3
│   └── transcript.txt
├── config.yaml
├── data/
│   └── test_sample5.mp4
├── testing.ipynb
└── tools/
    ├── chatbot.py
    ├── create_video_transcript.py
    ├── load_video.py
    └── utils/
        ├── embedding.py
        ├── vector_store.py
        └── chat_memory.py
```

## Required Implementations

### 1. Video Frame Processing Pipeline 🚧

**File**: `tools/video_frame_processor.py`
```python
# Features needed:
- Frame extraction at configurable intervals (1-2 fps for efficiency)
- VLM-based frame analysis and importance scoring
- Smart frame selection based on content analysis
- Frame caching with metadata (timestamp, importance, description)
- Support for different video types (traffic, podcasts, educational)
```

**Integration with**: 
- Existing transcript timestamps for alignment
- VLM models (Ollama vision capabilities)
- Cache directory structure

### 2. Object Detection & Tracking System 🚧

**File**: `tools/object_tracker.py`
```python
# Core functionality:
- YOLO/Detectron2 for object detection per frame
- DeepSORT/ByteTrack for persistent ID assignment
- Object trajectory tracking across video timeline  
- Bounding box coordinates and confidence scores
- Object metadata storage (type, description, actions)
```

**Object ID System**:
- Persistent IDs: car_1, car_2, person_1, person_2
- RGB color assignment for visualization
- Cross-chunk ID persistence for long videos
- Object re-identification across video segments

### 3. Graph Database Integration 🚧

**File**: `tools/utils/graph_store.py`
```python
# Database choice: Neo4j or NetworkX for prototyping
# Schema design:

# Node Types:
- VIDEO_NODE: {id, duration, type, metadata}
- FRAME_NODE: {timestamp, importance_score, description, objects}  
- TRANSCRIPT_SEGMENT: {start_time, end_time, text}
- OBJECT_NODE: {id, type, first_seen, last_seen}
- USER_SESSION: {session_id, timestamp, user_id}
- CHAT_MESSAGE: {content, timestamp, references}

# Relationship Types:
- TEMPORAL_NEXT: Frame A FOLLOWS Frame B
- CONTAINS: Video CONTAINS Frame/Transcript
- ALIGNS_WITH: Frame ALIGNS_WITH Transcript  
- TRACKS: Object TRACKS across multiple frames
- REFERENCES: Chat REFERENCES Object/Timestamp
- DISCUSSES: Session DISCUSSES Video_Moment
```

### 4. Enhanced VLM Integration 🚧

**Enhance**: `tools/chatbot.py`
```python
# Additional capabilities:
- Multi-frame context processing
- Object-aware image analysis
- Temporal reasoning across video segments
- Integration with tracking metadata
- Frame + transcript + object context fusion
```

**VLM Model Options**:
- **LLaVA** for detailed frame analysis
- **Ollama vision models** (llama3.2-vision, bakllava)
- **Qwen-VL** for long context handling
- Integration with existing Ollama setup

### 5. Data Pipeline Orchestrator 🚧

**File**: `tools/video_processor.py`
```python
# Main processing pipeline:
1. Video ingestion and audio extraction
2. Audio transcription with timestamps
3. Frame extraction and VLM analysis  
4. Object detection and tracking
5. Graph database population
6. Vector store indexing
7. Cache management and optimization

# Workflow coordination:
- Handle different video types intelligently
- Manage processing chunks for long videos
- Optimize for performance (parallel processing)
- Error handling and recovery
```

### 6. Comprehensive Memory System 🚧

**Enhanced**: `tools/utils/chat_memory.py`
```python
# Multi-level memory architecture:

# 1. Session Memory (Short-term)
- Current conversation context
- Active video/timestamp being discussed
- Recently mentioned object IDs
- User query history for current session

# 2. Video Context Memory (Medium-term)  
- Per-video conversation history
- Key moments previously discussed
- Object interactions that were highlighted
- User-specific insights per video

# 3. User Profile Memory (Long-term)
- Cross-session user preferences
- Frequently asked question patterns
- Domain expertise level (traffic/podcasts/educational)
- Personalization data
```

**Memory Integration Points**:
```python
# Graph database relationships:
USER_SESSION → DISCUSSES → VIDEO_MOMENT
USER_SESSION → REFERENCES → OBJECT_ID  
CHAT_MESSAGE → MENTIONS → TIMESTAMP
CHAT_MESSAGE → ASKS_ABOUT → OBJECT_NODE

# Vector store enhancements:
- Chat history embeddings for semantic search
- Conversation context retrieval
- "Remember when we talked about..." functionality
```

### 7. Enhanced Configuration Management 🚧

**Enhanced**: `config.yaml`
```yaml
# Video processing settings
video_processing:
  frame_extraction_fps: 1
  chunk_duration_seconds: 30
  overlap_seconds: 5
  supported_formats: [mp4, avi, mov, mkv]

# Object tracking settings  
object_tracking:
  detection_model: "yolo"
  tracking_algorithm: "deepsort"
  confidence_threshold: 0.5
  max_tracks_per_frame: 50

# VLM settings
vlm:
  model_name: "gemma3:4b"
  importance_threshold: 0.6
  batch_size: 4
  context_frames: 3

# Database settings
databases:
  vector_store:
    type: "milvus"
    uri: "./milvus_video.db"
  graph_store:
    type: "neo4j"
    uri: "bolt://localhost:7687"
    
# Memory settings
memory:
  session_retention_hours: 24
  max_conversation_history: 100
  enable_cross_session_memory: true
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Video frame processing pipeline
- [ ] Basic VLM integration for frame analysis
- [ ] Enhanced configuration system
- [ ] Frame caching and metadata storage

### Phase 2: Object Intelligence (Week 2-3)  
- [ ] Object detection integration (YOLO)
- [ ] Persistent tracking system (DeepSORT)
- [ ] Object ID management and visualization
- [ ] Trajectory storage and analysis

### Phase 3: Graph Intelligence (Week 3-4)
- [ ] Graph database setup (Neo4j/NetworkX)
- [ ] Schema implementation and relationships
- [ ] Temporal reasoning capabilities  
- [ ] Multi-hop query support

### Phase 4: Memory & Context (Week 4-5)
- [ ] Comprehensive memory system
- [ ] Session persistence and retrieval
- [ ] User profile management
- [ ] Context-aware responses

### Phase 5: Integration & Optimization (Week 5-6)
- [ ] End-to-end pipeline integration
- [ ] Performance optimization
- [ ] Error handling and edge cases
- [ ] Demo preparation and testing

## Technical Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│ Audio Extraction │───▶│  Transcription  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Frame Processor │───▶│  VLM Analysis    │───▶│ Importance Score│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│Object Detection │───▶│   ID Tracking    │───▶│  Graph Storage  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
         ┌─────────────────────────────────────────────────┐
         │                Vector Store                     │
         │  (Frame descriptions, object metadata,          │
         │   conversation embeddings)                      │  
         └─────────────────────────────────────────────────┘
                                 │
                                 ▼
         ┌─────────────────────────────────────────────────┐
         │              Chat Interface                     │
         │  (Multi-turn conversations with object          │
         │   referencing and temporal reasoning)           │
         └─────────────────────────────────────────────────┘
```

## Initial Chatbot Output

### Automatic Video Summary Generation
After video processing completion, the chatbot should immediately provide an LLM-generated summary using all preprocessed data:

**Processing Flow:**
1. **Video Processing Complete** → All data cached (frames, transcript, objects, graph relationships)
2. **LLM Summary Generation** → Feed preprocessed data to LLM:
   ```python
   # Input to LLM:
   - Full transcript with timestamps
   - Key frame descriptions with importance scores  
   - Object tracking data (if applicable)
   - Scene change points
   - Content type classification
   ```
3. **Auto-Generated Summary Output**:
   ```
   **Video Summary** (5:30 duration)
   
   This educational video explains the difference between "Did you" and "Have you" in English grammar. 
   
   **Key Moments:**
   • 00:15-00:45: Introduction to the main concept with examples
   • 01:20-02:10: Detailed explanation of "Did you" usage in simple past tense  
   • 02:45-03:30: "Have you" usage in present perfect tense with examples
   • 04:15-04:45: Interactive question for viewer engagement
   
   The speaker provides clear examples and asks viewers to practice in comments.
   ```
4. **Ready for Conversation** → User can then ask follow-up questions about specific moments

**Implementation**: `tools/video_summarizer.py`
- Compile all preprocessed data into LLM-friendly format
- Generate contextual summary with key moments and timestamps
- Correlate transcript segments with visual analysis
- Prepare system for follow-up conversations

## Key Features to Implement

### Object Referencing System
```python
# Enable chatbot responses like:
"Car 3 ran a red light at timestamp 2:15"
"Person 1 entered the frame at 1:30 and left at 2:45"
"Show me all instances where Car 3 appeared"
```

### Temporal Reasoning
```python
# Support queries like:
"What happened before the accident?"
"Show me the sequence of events leading to the violation"
"What did Person 1 do after picking up the object?"
```

### Multi-Domain Support
```python
# Adaptive analysis for different content types:
- Traffic monitoring: Violations, vehicle tracking, pedestrian analysis
- Podcasts: Key topics, speaker changes, important quotes  
- Educational: Concepts explained, visual aids, demonstration steps
```

### Performance Requirements
- **Latency**: <2 seconds for query responses (after initial processing)
- **Video Support**: 10-minute videos at 60fps
- **Scalability**: Handle multiple concurrent video analyses
- **Memory Efficiency**: Smart caching and cleanup strategies

## Dependencies to Add

```bash
# Computer vision
pip install opencv-python torch torchvision ultralytics

# Object tracking  
pip install deep-sort-realtime filterpy

# Graph database
pip install neo4j networkx

# Additional ML/AI
pip install transformers accelerate bitsandbytes

# Video processing enhancement
pip install ffmpeg-python pillow

# Memory management
pip install redis sqlite3 (built-in)
```

## Success Metrics

1. **Functional**: Object ID persistence across video timeline
2. **Performance**: Sub-2 second query response time
3. **Accuracy**: >90% object detection and tracking accuracy  
4. **Usability**: Natural language object referencing
5. **Memory**: Contextual conversation continuity
6. **Scalability**: Support for 10+ minute videos efficiently

## Future Enhancements

- Real-time video stream processing
- Multiple camera feed support  
- Advanced anomaly detection
- Custom object training capabilities
- API endpoints for integration
- Web interface for video upload and chat

---

**Note**: This implementation plan provides a comprehensive roadmap for building a sophisticated multimodal video chat assistant with object tracking, temporal reasoning, and persistent memory capabilities.