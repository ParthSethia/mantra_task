import os
import json
import hashlib
import yaml
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path

class VideoCacheManager:
    """Manages video processing cache to avoid reprocessing"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.cache_config = config['cache']
        self.cache_enabled = self.cache_config.get('enable_video_cache', True)
        self.validation_method = self.cache_config.get('cache_validation', 'checksum')
        self.max_age_hours = self.cache_config.get('max_cache_age_hours', 168)  # 7 days
        
        # Cache index file
        self.index_file = self.cache_config.get('cache_index_file', 'cache/video_cache_index.json')
        self.cache_directory = self.cache_config['directory']
        
        # Ensure cache directories exist
        os.makedirs(self.cache_directory, exist_ok=True)
        os.makedirs(self.cache_config['metadata_directory'], exist_ok=True)
        
        # Load cache index
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict:
        """Load the video cache index"""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Failed to load cache index: {e}")
            return {}
    
    def _save_cache_index(self):
        """Save the video cache index"""
        try:
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            with open(self.index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            print(f"Failed to save cache index: {e}")
    
    def _get_video_checksum(self, video_path: str) -> str:
        """Calculate MD5 checksum of video file"""
        hash_md5 = hashlib.md5()
        try:
            with open(video_path, "rb") as f:
                # Read file in chunks to handle large videos
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Failed to calculate checksum: {e}")
            return ""
    
    def _get_video_modified_time(self, video_path: str) -> str:
        """Get video file modification time"""
        try:
            return str(os.path.getmtime(video_path))
        except Exception as e:
            print(f"Failed to get modification time: {e}")
            return ""
    
    def _get_cache_key(self, video_path: str) -> str:
        """Generate cache key for video"""
        # Use absolute path to ensure uniqueness
        abs_path = os.path.abspath(video_path)
        return hashlib.md5(abs_path.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, video_path: str, cache_entry: Dict) -> bool:
        """Check if cached data is still valid"""
        try:
            # Check if cache is enabled
            if not self.cache_enabled:
                return False
            
            # Check cache age
            processed_time = datetime.fromisoformat(cache_entry.get('processed_at', ''))
            max_age = timedelta(hours=self.max_age_hours)
            if datetime.now() - processed_time > max_age:
                print(f"Cache expired (age: {datetime.now() - processed_time})")
                return False
            
            # Check if video file still exists
            if not os.path.exists(video_path):
                print("Original video file no longer exists")
                return False
            
            # Validate based on configured method
            if self.validation_method == 'checksum':
                current_checksum = self._get_video_checksum(video_path)
                cached_checksum = cache_entry.get('checksum', '')
                if current_checksum != cached_checksum:
                    print(f"Video checksum changed: {current_checksum} != {cached_checksum}")
                    return False
            
            elif self.validation_method == 'modified_time':
                current_mtime = self._get_video_modified_time(video_path)
                cached_mtime = cache_entry.get('modified_time', '')
                if current_mtime != cached_mtime:
                    print(f"Video modification time changed: {current_mtime} != {cached_mtime}")
                    return False
            
            # Check if all required cache files exist
            required_files = [
                cache_entry.get('metadata_file'),
                cache_entry.get('graph_file'),
                cache_entry.get('frame_analysis_file')
            ]
            
            for file_path in required_files:
                if file_path and not os.path.exists(file_path):
                    print(f"Required cache file missing: {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Cache validation error: {e}")
            return False
    
    def get_cached_data(self, video_path: str) -> Optional[Dict]:
        """Get cached video processing data if available and valid"""
        if not self.cache_enabled:
            return None
        
        cache_key = self._get_cache_key(video_path)
        
        if cache_key not in self.cache_index:
            print("Video not found in cache")
            return None
        
        cache_entry = self.cache_index[cache_key]
        
        if not self._is_cache_valid(video_path, cache_entry):
            print("Cache invalid, will reprocess")
            # Remove invalid cache entry
            del self.cache_index[cache_key]
            self._save_cache_index()
            return None
        
        try:
            print("Loading video data from cache...")
            
            # Load cached metadata
            metadata_file = cache_entry.get('metadata_file')
            if metadata_file and os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Add cache info
                cached_data['cache_info'] = {
                    'loaded_from_cache': True,
                    'processed_at': cache_entry.get('processed_at'),
                    'cache_key': cache_key
                }
                
                print(f"✓ Video data loaded from cache (processed: {cache_entry.get('processed_at')})")
                return cached_data
            else:
                print("Cache metadata file not found")
                return None
                
        except Exception as e:
            print(f"Failed to load cached data: {e}")
            return None
    
    def cache_video_data(self, video_path: str, processing_results: Dict, consolidated_data: Dict):
        """Cache video processing results"""
        if not self.cache_enabled:
            return
        
        try:
            cache_key = self._get_cache_key(video_path)
            
            # Create cache entry
            cache_entry = {
                'video_path': os.path.abspath(video_path),
                'processed_at': datetime.now().isoformat(),
                'processing_results': processing_results,
                'cache_key': cache_key
            }
            
            # Add validation data
            if self.validation_method == 'checksum':
                cache_entry['checksum'] = self._get_video_checksum(video_path)
            elif self.validation_method == 'modified_time':
                cache_entry['modified_time'] = self._get_video_modified_time(video_path)
            
            # Store file paths
            metadata_file = os.path.join(self.cache_config['metadata_directory'], f'video_metadata_{cache_key}.json')
            cache_entry['metadata_file'] = metadata_file
            cache_entry['graph_file'] = os.path.join(self.cache_config['metadata_directory'], 'knowledge_graph.json')
            cache_entry['frame_analysis_file'] = os.path.join(self.cache_config['metadata_directory'], 'frame_analysis.json')
            cache_entry['tracking_file'] = os.path.join(self.cache_config['metadata_directory'], 'object_tracking.json')
            
            # Save consolidated metadata with cache key
            with open(metadata_file, 'w') as f:
                json.dump(consolidated_data, f, indent=2)
            
            # Update cache index
            self.cache_index[cache_key] = cache_entry
            self._save_cache_index()
            
            print(f"✓ Video data cached with key: {cache_key}")
            
        except Exception as e:
            print(f"Failed to cache video data: {e}")
    
    def invalidate_cache(self, video_path: str = None):
        """Invalidate cache for specific video or all videos"""
        if video_path:
            # Invalidate specific video
            cache_key = self._get_cache_key(video_path)
            if cache_key in self.cache_index:
                del self.cache_index[cache_key]
                print(f"Cache invalidated for: {video_path}")
        else:
            # Invalidate all cache
            self.cache_index.clear()
            print("All video cache invalidated")
        
        self._save_cache_index()
    
    def cleanup_cache(self):
        """Remove expired and invalid cache entries"""
        print("Cleaning up video cache...")
        
        keys_to_remove = []
        for cache_key, cache_entry in self.cache_index.items():
            video_path = cache_entry.get('video_path', '')
            
            if not video_path or not self._is_cache_valid(video_path, cache_entry):
                keys_to_remove.append(cache_key)
                
                # Remove associated files
                for file_key in ['metadata_file', 'graph_file', 'frame_analysis_file', 'tracking_file']:
                    file_path = cache_entry.get(file_key)
                    if file_path and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Failed to remove cache file {file_path}: {e}")
        
        # Remove invalid entries from index
        for key in keys_to_remove:
            del self.cache_index[key]
        
        if keys_to_remove:
            self._save_cache_index()
            print(f"Removed {len(keys_to_remove)} invalid cache entries")
        else:
            print("No cache entries to clean up")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_entries = len(self.cache_index)
        valid_entries = 0
        total_size = 0
        
        for cache_key, cache_entry in self.cache_index.items():
            video_path = cache_entry.get('video_path', '')
            if video_path and self._is_cache_valid(video_path, cache_entry):
                valid_entries += 1
                
                # Calculate cache size
                for file_key in ['metadata_file', 'graph_file', 'frame_analysis_file', 'tracking_file']:
                    file_path = cache_entry.get(file_key)
                    if file_path and os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        
        return {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'invalid_entries': total_entries - valid_entries,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_enabled': self.cache_enabled,
            'validation_method': self.validation_method,
            'max_age_hours': self.max_age_hours
        }


if __name__ == "__main__":
    # Test video cache manager
    cache_manager = VideoCacheManager()
    
    print("Video Cache Manager")
    print("=" * 30)
    
    stats = cache_manager.get_cache_stats()
    print(f"Cache entries: {stats['valid_entries']}/{stats['total_entries']}")
    print(f"Cache size: {stats['total_size_mb']:.1f} MB")
    print(f"Validation: {stats['validation_method']}")
    print(f"Max age: {stats['max_age_hours']} hours")
    
    # Cleanup old entries
    cache_manager.cleanup_cache()