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
            texts.append(frame_data['vlm_analysis'])
            metadatas.append({
                'frame_id': frame_data['frame_id'],
                'timestamp': frame_data['timestamp'],
                'importance_score': frame_data['importance_score'],
                'frame_path': frame_data['frame_path'],
                'type': 'frame_analysis'
            })
        
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    def add_transcript_segments(self, transcript_segments):
        """Add transcript segments to vector store"""
        texts = []
        metadatas = []
        
        for segment in transcript_segments:
            texts.append(segment['text'])
            metadatas.append({
                'start_time': segment['start'],
                'end_time': segment['end'],
                'type': 'transcript_segment'
            })
        
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    def search_by_query(self, query: str, k: int = 5):
        """Search for relevant content by query"""
        return self.vector_store.similarity_search(query, k=k)
    
    def search_frames_by_timestamp(self, timestamp: float, window: float = 30.0, k: int = 3):
        """Search for frames within a time window"""
        # This would require custom filtering - simplified for now
        return self.vector_store.similarity_search(f"timestamp around {timestamp}", k=k)