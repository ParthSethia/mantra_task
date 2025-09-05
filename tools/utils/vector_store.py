import yaml
from langchain_milvus import Milvus
from .embedding import embeddings

class VideoVectorStore:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.vector_config = config['vector_store']
        
        self.vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": self.vector_config['uri']},
            index_params={
                "index_type": self.vector_config['index_type'], 
                "metric_type": self.vector_config['metric_type']
            },
        )
    
    def add_frame_descriptions(self, frame_data_list):
        """Add frame descriptions and metadata to vector store"""
        texts = []
        metadatas = []
        
        for frame_data in frame_data_list:
            # Clean and limit text for vector store compatibility
            raw_text = frame_data['vlm_analysis']
            cleaned_text = self._clean_text_for_vector_store(raw_text)
            
            texts.append(cleaned_text)
            metadatas.append({
                'frame_id': frame_data['frame_id'],
                'timestamp': frame_data['timestamp'],
                'start_time': frame_data['timestamp'],  # Use timestamp as start_time for frame data
                'end_time': frame_data['timestamp'],    # For frame data, start and end are the same
                'importance_score': frame_data['importance_score'],
                'frame_path': frame_data['frame_path'],
                'type': 'frame_analysis'
            })
        
        try:
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        except Exception as e:
            print(f"Failed to add frame descriptions to vector store: {e}")
            # Try adding one by one to identify problematic entries
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                try:
                    self.vector_store.add_texts(texts=[text], metadatas=[metadata])
                except Exception as single_error:
                    print(f"Failed to add frame {metadata['frame_id']}: {single_error}")
                    print(f"Problematic text: {text[:200]}...")
    
    def _clean_text_for_vector_store(self, text: str, max_length: int = 8000) -> str:
        """Clean and format text for vector store compatibility"""
        if not text:
            return "No analysis available"
        
        # Convert to string if not already
        text = str(text)
        
        # Clean up Florence-2 specific formatting
        # Remove dictionary-like structures: {'<DETAILED_CAPTION>': '...'}
        import re
        
        # Extract content from dictionary-like structures
        dict_pattern = r"\{'[^']+': '([^']+)'\}"
        matches = re.findall(dict_pattern, text)
        if matches:
            # Use the first extracted content
            text = matches[0]
        
        # Clean up common problematic patterns
        text = re.sub(r'\{[^}]*\}', '', text)  # Remove any remaining dictionary structures
        text = re.sub(r'<[^>]*>', '', text)    # Remove XML-like tags
        text = text.replace('\n', ' ')          # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)       # Normalize whitespace
        text = text.strip()
        
        # Limit text length
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Ensure text is not empty
        if not text or text.isspace():
            text = "Frame analysis content not available"
        
        return text
    
    def add_transcript_segments(self, transcript_segments):
        """Add transcript segments to vector store"""
        texts = []
        metadatas = []
        
        for segment in transcript_segments:
            # Clean transcript text
            cleaned_text = self._clean_text_for_vector_store(segment['text'], max_length=4000)
            texts.append(cleaned_text)
            metadatas.append({
                'start_time': segment['start'],
                'end_time': segment['end'],
                'type': 'transcript_segment'
            })
        
        try:
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        except Exception as e:
            print(f"Failed to add transcript segments to vector store: {e}")
            # Try adding one by one to identify problematic entries
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                try:
                    self.vector_store.add_texts(texts=[text], metadatas=[metadata])
                except Exception as single_error:
                    print(f"Failed to add transcript segment {i}: {single_error}")
                    print(f"Problematic text: {text[:200]}...")
    
    def search_by_query(self, query: str, k: int = 5):
        """Search for relevant content by query"""
        return self.vector_store.similarity_search(query, k=k)
    
    def search_frames_by_timestamp(self, timestamp: float, window: float = 30.0, k: int = 3):
        """Search for frames within a time window"""
        # This would require custom filtering - simplified for now
        return self.vector_store.similarity_search(f"timestamp around {timestamp}", k=k)