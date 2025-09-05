import cv2
import os
import json
import yaml
from PIL import Image
import base64
from io import BytesIO
from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from hf_models import HuggingFaceVLM

class VideoFrameProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.video_config = self.config['video_processing']
        self.vlm_config = self.config['vlm']
        self.cache_config = self.config['cache']
        
        # Initialize VLM based on model type
        self.model_type = self.vlm_config.get('model_type', 'ollama')
        
        if self.model_type == 'ollama':
            self.llm = ChatOllama(
                model=self.vlm_config['model_name'], 
                temperature=self.vlm_config['temperature']
            )
            self.chain = self._create_analysis_chain()
            self.hf_vlm = None
        else:
            self.llm = None
            self.chain = None
            self.hf_vlm = HuggingFaceVLM(config_path)
        
        # Create cache directories
        os.makedirs(self.cache_config['frames_directory'], exist_ok=True)
        os.makedirs(self.cache_config['metadata_directory'], exist_ok=True)
    
    def _create_analysis_chain(self):
        def prompt_func(data):
            text = data["text"]
            image = data["image"]
            
            image_part = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image}",
            }
            
            text_part = {"type": "text", "text": text}
            content_parts = [image_part, text_part]
            
            return [HumanMessage(content=content_parts)]
        
        return prompt_func | self.llm | StrOutputParser()
    
    def convert_to_base64(self, pil_image: Image.Image) -> str:
        if pil_image is None:
            raise ValueError("PIL image is None, cannot convert to base64")
            
        try:
            # Resize image for faster processing
            target_size = tuple(self.video_config.get('resize_dimensions', [640, 480]))
            resized_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            
            buffered = BytesIO()
            quality = self.video_config.get('jpeg_quality', 85)
            resized_image.save(buffered, format="JPEG", quality=quality)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str
            
        except Exception as e:
            raise ValueError(f"Failed to convert image to base64: {str(e)}")
    
    def extract_frames(self, video_path: str) -> List[Dict]:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / self.video_config['frame_extraction_fps'])
        
        frames_data = []
        frame_count = 0
        saved_frame_count = 0
        
        print(f"Extracting frames at {self.video_config['frame_extraction_fps']} fps from video with {fps} fps")
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            # Validate frame data
            if frame is None:
                print(f"Warning: Frame {frame_count} is None, skipping...")
                frame_count += 1
                continue
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                try:
                    # Convert frame to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Save frame
                    frame_filename = f"frame_{saved_frame_count:06d}_{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(self.cache_config['frames_directory'], frame_filename)
                    pil_image.save(frame_path, "JPEG")
                    
                    # Store frame metadata
                    try:
                        base64_image = self.convert_to_base64(pil_image)
                        frame_data = {
                            'frame_id': saved_frame_count,
                            'timestamp': timestamp,
                            'frame_path': frame_path,
                            'pil_image': pil_image,
                            'base64_image': base64_image
                        }
                        frames_data.append(frame_data)
                        saved_frame_count += 1
                    except ValueError as e:
                        print(f"Error creating base64 for frame {frame_count}: {e}")
                        continue
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count} at timestamp {timestamp:.2f}: {e}")
                    frame_count += 1
                    continue
                
                if saved_frame_count % 10 == 0:
                    print(f"Extracted {saved_frame_count} frames...")
            
            frame_count += 1
        
        video.release()
        print(f"Total frames extracted: {saved_frame_count}")
        return frames_data
    
    def analyze_frames_batch(self, frames_batch: List[Dict]) -> List[Dict]:
        """Analyze multiple frames in a single batch request"""
        if not frames_batch:
            return []
        
        try:
            if self.model_type == 'ollama':
                return self._analyze_batch_ollama(frames_batch)
            else:
                return self._analyze_batch_huggingface(frames_batch)
                
        except Exception as e:
            print(f"Batch analysis failed: {e}. Falling back to individual analysis.")
            return [self.analyze_frame_importance_single(frame) for frame in frames_batch]
    
    def _analyze_batch_ollama(self, frames_batch: List[Dict]) -> List[Dict]:
        """Batch analysis using Ollama"""
        batch_prompt = """Analyze these video frames and provide analysis for EACH frame. 
For each frame, provide:
1. Brief description (1-2 sentences)
2. Importance score (0.0-1.0)
3. Key objects/elements

Format response as:
FRAME_1:
DESCRIPTION: [description]
IMPORTANCE: [0.0-1.0]
OBJECTS: [objects]

FRAME_2:
DESCRIPTION: [description]
IMPORTANCE: [0.0-1.0]
OBJECTS: [objects]

Continue for all frames..."""

        # Create content parts with all images
        content_parts = []
        content_parts.append({"type": "text", "text": batch_prompt})
        
        for i, frame_data in enumerate(frames_batch):
            content_parts.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{frame_data['base64_image']}"
            })
            content_parts.append({
                "type": "text", 
                "text": f"Frame {i+1} (timestamp: {frame_data['timestamp']:.2f}s):"
            })
        
        # Get batch response
        message = [HumanMessage(content=content_parts)]
        response = self.llm.invoke(message).content
        
        # Parse batch response
        return self._parse_batch_response(response, frames_batch)
    
    def _analyze_batch_huggingface(self, frames_batch: List[Dict]) -> List[Dict]:
        """Batch analysis using HuggingFace models"""
        analyzed_frames = []
        
        # Extract PIL images from batch, loading from file if needed
        images = []
        for frame_data in frames_batch:
            if frame_data.get('pil_image') is not None:
                images.append(frame_data['pil_image'])
            elif 'frame_path' in frame_data and os.path.exists(frame_data['frame_path']):
                # Reload image from file if PIL image is missing
                try:
                    pil_image = Image.open(frame_data['frame_path'])
                    images.append(pil_image)
                    frame_data['pil_image'] = pil_image  # Cache for future use
                except Exception as e:
                    print(f"Failed to load frame from {frame_data['frame_path']}: {e}")
                    images.append(None)
            else:
                print(f"No image data available for frame {frame_data.get('frame_id', 'unknown')}")
                images.append(None)
        
        # Get captions from HF model
        captions = self.hf_vlm.analyze_batch(images)
        
        # Convert to analysis format
        for i, frame_data in enumerate(frames_batch):
            caption = captions[i] if i < len(captions) else "Analysis failed"
            
            # Simple importance scoring based on caption length and content
            importance_score = self._score_caption_importance(caption)
            
            analyzed_frames.append({
                'frame_id': frame_data['frame_id'],
                'timestamp': frame_data['timestamp'],
                'frame_path': frame_data['frame_path'],
                'vlm_analysis': f"DESCRIPTION: {caption}\nIMPORTANCE: {importance_score:.2f}",
                'importance_score': importance_score,
                'analyzed': True
            })
        
        return analyzed_frames
    
    def _score_caption_importance(self, caption: str) -> float:
        """Simple importance scoring for HF model captions"""
        if not caption or "error" in caption.lower() or "failed" in caption.lower():
            return 0.3
        
        # Basic scoring heuristics
        score = 0.5  # Base score
        
        # Longer descriptions might be more detailed/important
        if len(caption) > 50:
            score += 0.1
        
        # Presence of action words
        action_words = ['person', 'people', 'walking', 'driving', 'running', 'moving', 'holding']
        for word in action_words:
            if word in caption.lower():
                score += 0.1
                break
        
        # Presence of multiple objects
        object_words = ['car', 'truck', 'bike', 'building', 'tree', 'sign', 'road']
        object_count = sum(1 for word in object_words if word in caption.lower())
        score += min(object_count * 0.05, 0.2)
        
        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _calculate_importance_from_caption(self, caption: str) -> float:
        """Calculate importance score from caption text (for HuggingFace models)"""
        if not caption or "Analysis failed" in caption:
            return 0.3  # Default low score for failed analysis
        
        score = 0.3  # Base score
        
        # Length-based scoring (longer captions tend to be more detailed)
        if len(caption) > 100:
            score += 0.2
        elif len(caption) > 50:
            score += 0.1
        
        # Content-based scoring
        important_words = [
            # People and activities
            'person', 'people', 'man', 'woman', 'child', 'walking', 'running', 'sitting',
            # Vehicles and movement
            'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'driving', 'moving', 'parked',
            # Actions and events
            'holding', 'carrying', 'using', 'working', 'talking', 'meeting', 'event',
            # Important objects
            'sign', 'building', 'door', 'window', 'computer', 'phone', 'book', 'tool',
            # Scene indicators
            'crowded', 'busy', 'active', 'empty', 'bright', 'dark', 'indoor', 'outdoor'
        ]
        
        caption_lower = caption.lower()
        word_matches = sum(1 for word in important_words if word in caption_lower)
        score += min(word_matches * 0.05, 0.3)  # Up to 0.3 bonus for important words
        
        # Activity indicators (higher importance)
        activity_words = ['moving', 'action', 'event', 'happening', 'activity', 'busy', 'active']
        if any(word in caption_lower for word in activity_words):
            score += 0.1
        
        # Multiple objects increase importance
        if caption_lower.count('and') >= 2:  # Multiple items connected by 'and'
            score += 0.1
        
        return min(max(score, 0.1), 1.0)  # Clamp between 0.1 and 1.0
    
    def analyze_frame_importance_single(self, frame_data: Dict) -> Dict:
        try:
            if self.model_type == 'ollama':
                # Use Ollama with LangChain for detailed analysis
                response = self._analyze_with_ollama(frame_data)
                importance_score = self._extract_importance_score(response)
            else:
                # Use HuggingFace models (BLIP/Florence-2)
                response = self._analyze_with_huggingface(frame_data)
                importance_score = self._calculate_importance_from_caption(response)
            
            analysis = {
                'frame_id': frame_data['frame_id'],
                'timestamp': frame_data['timestamp'],
                'frame_path': frame_data['frame_path'],
                'vlm_analysis': response,
                'importance_score': importance_score,
                'analyzed': True
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing frame {frame_data['frame_id']}: {e}")
            return {
                'frame_id': frame_data['frame_id'],
                'timestamp': frame_data['timestamp'],
                'frame_path': frame_data['frame_path'],
                'vlm_analysis': f"Analysis failed: {str(e)}",
                'importance_score': 0.5,  # Default score
                'analyzed': False
            }
    
    def _analyze_with_ollama(self, frame_data: Dict) -> str:
        """Analyze frame using Ollama with detailed prompt"""
        if self.chain is None:
            raise ValueError("Ollama chain not initialized")
            
        analysis_prompt = """Analyze this video frame and provide:
1. A brief description of what's happening (2-3 sentences)
2. Importance score (0.0-1.0) based on:
   - Visual activity/motion
   - Presence of people or objects
   - Scene changes or significant events
   - Educational or informational content value
3. Key objects or elements visible
4. Any text or important visual information

Format your response as:
DESCRIPTION: [your description]
IMPORTANCE: [score from 0.0 to 1.0]
OBJECTS: [list of key objects/elements]
DETAILS: [any additional important details]"""

        return self.chain.invoke({
            "text": analysis_prompt,
            "image": frame_data['base64_image']
        })
    
    def _analyze_with_huggingface(self, frame_data: Dict) -> str:
        """Analyze frame using HuggingFace models (BLIP/Florence-2)"""
        if self.hf_vlm is None:
            raise ValueError("HuggingFace VLM not initialized")
        
        # Get PIL image, reload from file if needed
        pil_image = frame_data.get('pil_image')
        
        if pil_image is None and 'frame_path' in frame_data:
            frame_path = frame_data['frame_path']
            if os.path.exists(frame_path):
                try:
                    pil_image = Image.open(frame_path)
                    frame_data['pil_image'] = pil_image  # Cache it
                except Exception as e:
                    raise ValueError(f"Failed to load image from {frame_path}: {e}")
            else:
                raise ValueError(f"Frame path does not exist: {frame_path}")
        
        if pil_image is None:
            raise ValueError("No PIL image available for analysis")
        
        # Get caption from HuggingFace model
        caption = self.hf_vlm.analyze_image(pil_image)
        
        # Format as structured response similar to Ollama
        formatted_response = f"DESCRIPTION: {caption}\nIMPORTANCE: [calculated from content]\nOBJECTS: [detected from description]\nDETAILS: Frame analyzed using {self.model_type} model"
        
        return formatted_response
    
    def _analyze_batch_ollama(self, frames_batch: List[Dict]) -> List[Dict]:
        """Batch analysis using Ollama"""
        if self.chain is None:
            raise ValueError("Ollama chain not initialized")
            
        batch_prompt = """Analyze these video frames and provide analysis for EACH frame. 
For each frame, provide:
1. Brief description (1-2 sentences)
2. Importance score (0.0-1.0)
3. Key objects/elements

Format response as:
FRAME_1:
DESCRIPTION: [description]
IMPORTANCE: [0.0-1.0]
OBJECTS: [objects]

FRAME_2:
DESCRIPTION: [description]
IMPORTANCE: [0.0-1.0]
OBJECTS: [objects]

And so on for each frame..."""

        # Build content with all frame images
        content_parts = []
        
        # Add text prompt
        content_parts.append({"type": "text", "text": batch_prompt})
        
        # Add each frame image
        for i, frame_data in enumerate(frames_batch):
            if 'base64_image' in frame_data and frame_data['base64_image']:
                content_parts.append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{frame_data['base64_image']}"
                })
        
        try:
            # Create message with all images
            from langchain_core.messages import HumanMessage
            message = HumanMessage(content=content_parts)
            response = self.llm.invoke([message])
            
            # Parse batch response
            return self._parse_batch_response(response.content, frames_batch)
            
        except Exception as e:
            print(f"Ollama batch analysis failed: {e}")
            # Fallback to individual analysis
            return [self.analyze_frame_importance_single(frame) for frame in frames_batch]
    
    def _analyze_batch_huggingface(self, frames_batch: List[Dict]) -> List[Dict]:
        """Batch analysis using HuggingFace models"""
        if self.hf_vlm is None:
            raise ValueError("HuggingFace VLM not initialized")
            
        analyzed_frames = []
        
        # Extract PIL images from batch, loading from file if needed
        images = []
        for frame_data in frames_batch:
            if frame_data.get('pil_image') is not None:
                images.append(frame_data['pil_image'])
            elif 'frame_path' in frame_data and os.path.exists(frame_data['frame_path']):
                # Reload image from file if PIL image is missing
                try:
                    pil_image = Image.open(frame_data['frame_path'])
                    images.append(pil_image)
                    frame_data['pil_image'] = pil_image  # Cache for future use
                except Exception as e:
                    print(f"Failed to load frame from {frame_data['frame_path']}: {e}")
                    images.append(None)
            else:
                print(f"No image data available for frame {frame_data.get('frame_id', 'unknown')}")
                images.append(None)
        
        # Get captions from HF model
        captions = self.hf_vlm.analyze_batch(images)
        
        # Convert to analysis format
        for i, frame_data in enumerate(frames_batch):
            caption = captions[i] if i < len(captions) else "Analysis failed"
            
            # Simple importance scoring based on caption length and content
            importance_score = self._calculate_importance_from_caption(caption)
            
            analyzed_frames.append({
                'frame_id': frame_data['frame_id'],
                'timestamp': frame_data['timestamp'],
                'frame_path': frame_data['frame_path'],
                'vlm_analysis': f"DESCRIPTION: {caption}\nIMPORTANCE: {importance_score:.2f}\nDETAILS: Frame analyzed using {self.model_type} model",
                'importance_score': importance_score,
                'analyzed': True
            })
        
        return analyzed_frames
    
    def _parse_batch_response(self, response: str, frames_batch: List[Dict]) -> List[Dict]:
        """Parse batch analysis response into individual frame analyses"""
        analyzed_frames = []
        
        # Split response by frame markers
        frame_sections = response.split('FRAME_')
        
        for i, frame_data in enumerate(frames_batch):
            try:
                # Find corresponding section (i+1 because we split on 'FRAME_')
                section_idx = i + 1
                if section_idx < len(frame_sections):
                    section = frame_sections[section_idx]
                    
                    # Extract description, importance, objects
                    description = self._extract_field_from_section(section, 'DESCRIPTION')
                    importance_text = self._extract_field_from_section(section, 'IMPORTANCE')
                    objects_text = self._extract_field_from_section(section, 'OBJECTS')
                    
                    # Parse importance score
                    importance_score = self._extract_importance_from_text(importance_text)
                    
                    # Combine analysis
                    full_analysis = f"DESCRIPTION: {description}\nIMPORTANCE: {importance_score}\nOBJECTS: {objects_text}"
                    
                    analyzed_frames.append({
                        'frame_id': frame_data['frame_id'],
                        'timestamp': frame_data['timestamp'],
                        'frame_path': frame_data['frame_path'],
                        'vlm_analysis': full_analysis,
                        'importance_score': importance_score,
                        'analyzed': True
                    })
                else:
                    # Fallback for missing sections
                    analyzed_frames.append(self._create_fallback_analysis(frame_data))
                    
            except Exception as e:
                print(f"Error parsing frame {i}: {e}")
                analyzed_frames.append(self._create_fallback_analysis(frame_data))
        
        return analyzed_frames
    
    def _extract_field_from_section(self, section: str, field_name: str) -> str:
        """Extract a field value from a section of text"""
        try:
            lines = section.split('\n')
            for line in lines:
                if field_name + ':' in line.upper():
                    return line.split(':', 1)[1].strip()
            return f"Not found ({field_name})"
        except:
            return f"Parse error ({field_name})"
    
    def _extract_importance_from_text(self, text: str) -> float:
        """Extract importance score from text"""
        try:
            import re
            numbers = re.findall(r'\d*\.?\d+', text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            return 0.5
        except:
            return 0.5
    
    def _create_fallback_analysis(self, frame_data: Dict) -> Dict:
        """Create fallback analysis for failed parsing"""
        return {
            'frame_id': frame_data['frame_id'],
            'timestamp': frame_data['timestamp'],
            'frame_path': frame_data['frame_path'],
            'vlm_analysis': 'Analysis failed - processed in batch mode',
            'importance_score': 0.5,
            'analyzed': False
        }
    
    def _extract_importance_score(self, analysis_text: str) -> float:
        try:
            lines = analysis_text.split('\n')
            for line in lines:
                if 'IMPORTANCE:' in line.upper():
                    score_text = line.split(':')[1].strip()
                    # Extract first number found
                    import re
                    numbers = re.findall(r'\d*\.?\d+', score_text)
                    if numbers:
                        score = float(numbers[0])
                        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            return 0.5  # Default if no score found
        except:
            return 0.5  # Default on any error
    
    def process_video_frames(self, video_path: str) -> List[Dict]:
        print(f"Processing frames from video: {video_path}")
        
        # Generate video ID for video-specific caching
        from create_video_transcript import get_video_id
        video_id = get_video_id(video_path)
        
        # Check if we have cached frame analysis for this specific video
        metadata_file = os.path.join(self.cache_config['metadata_directory'], f'frame_analysis_{video_id}.json')
        if os.path.exists(metadata_file):
            print(f"Found existing frame analysis for video {video_id}, loading from cache...")
            try:
                with open(metadata_file, 'r') as f:
                    cached_frames = json.load(f)
                print(f"Loaded {len(cached_frames)} analyzed frames from cache")
                return cached_frames
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading cached frame analysis: {e}, reprocessing...")
        
        # Extract frames
        frames_data = self.extract_frames(video_path)
        
        # Process frames in batches
        batch_size = self.vlm_config.get('batch_size', 4)
        analyzed_frames = []
        
        total_batches = (len(frames_data) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(frames_data), batch_size):
            batch_end = min(batch_idx + batch_size, len(frames_data))
            frames_batch = frames_data[batch_idx:batch_end]
            
            current_batch = (batch_idx // batch_size) + 1
            print(f"Analyzing batch {current_batch}/{total_batches} ({len(frames_batch)} frames)")
            
            # Analyze batch
            batch_results = self.analyze_frames_batch(frames_batch)
            analyzed_frames.extend(batch_results)
        
        # Save metadata with video-specific filename
        with open(metadata_file, 'w') as f:
            # Remove PIL image object for JSON serialization
            serializable_data = []
            for frame in analyzed_frames:
                frame_copy = frame.copy()
                serializable_data.append(frame_copy)
            
            json.dump(serializable_data, f, indent=2)
        
        print(f"Frame analysis completed and saved to: {metadata_file}")
        return analyzed_frames
    
    def get_important_frames(self, analyzed_frames: List[Dict], threshold: float = None) -> List[Dict]:
        if threshold is None:
            threshold = self.vlm_config['importance_threshold']
        
        important_frames = [frame for frame in analyzed_frames 
                          if frame['importance_score'] >= threshold]
        
        important_frames.sort(key=lambda x: x['importance_score'], reverse=True)
        return important_frames


if __name__ == "__main__":
    processor = VideoFrameProcessor()
    
    # Process the default video
    video_path = processor.video_config['input_path']
    if os.path.exists(video_path):
        analyzed_frames = processor.process_video_frames(video_path)
        important_frames = processor.get_important_frames(analyzed_frames)
        
        print(f"\nFound {len(important_frames)} important frames (threshold: {processor.vlm_config['importance_threshold']})")
        for frame in important_frames[:5]:  # Show top 5
            print(f"Frame {frame['frame_id']} at {frame['timestamp']:.2f}s - Score: {frame['importance_score']:.2f}")
    else:
        print(f"Video file not found: {video_path}")