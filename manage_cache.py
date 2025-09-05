#!/usr/bin/env python3
"""
Video Cache Management Utility
"""

import sys
import argparse
from tools.utils.video_cache import VideoCacheManager

def show_cache_stats(cache_manager):
    """Display cache statistics"""
    stats = cache_manager.get_cache_stats()
    
    print("📊 VIDEO CACHE STATISTICS")
    print("=" * 40)
    print(f"Cache Status: {'Enabled' if stats['cache_enabled'] else 'Disabled'}")
    print(f"Validation Method: {stats['validation_method']}")
    print(f"Max Age: {stats['max_age_hours']} hours")
    print(f"Total Entries: {stats['total_entries']}")
    print(f"Valid Entries: {stats['valid_entries']}")
    print(f"Invalid Entries: {stats['invalid_entries']}")
    print(f"Total Size: {stats['total_size_mb']:.1f} MB")

def cleanup_cache(cache_manager):
    """Clean up invalid cache entries"""
    print("🧹 CLEANING UP CACHE")
    print("=" * 30)
    cache_manager.cleanup_cache()

def invalidate_cache(cache_manager, video_path=None):
    """Invalidate cache entries"""
    if video_path:
        print(f"🗑️  INVALIDATING CACHE FOR: {video_path}")
        cache_manager.invalidate_cache(video_path)
    else:
        print("🗑️  INVALIDATING ALL CACHE")
        cache_manager.invalidate_cache()

def list_cached_videos(cache_manager):
    """List all cached videos"""
    print("📹 CACHED VIDEOS")
    print("=" * 40)
    
    if not cache_manager.cache_index:
        print("No videos in cache")
        return
    
    for cache_key, cache_entry in cache_manager.cache_index.items():
        video_path = cache_entry.get('video_path', 'Unknown')
        processed_at = cache_entry.get('processed_at', 'Unknown')
        is_valid = cache_manager._is_cache_valid(video_path, cache_entry)
        
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"{status} | {video_path}")
        print(f"         Processed: {processed_at}")
        print(f"         Cache Key: {cache_key}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Manage video processing cache")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Stats command
    subparsers.add_parser("stats", help="Show cache statistics")
    
    # List command
    subparsers.add_parser("list", help="List cached videos")
    
    # Cleanup command
    subparsers.add_parser("cleanup", help="Clean up invalid cache entries")
    
    # Invalidate command
    invalidate_parser = subparsers.add_parser("invalidate", help="Invalidate cache")
    invalidate_parser.add_argument("--video", help="Specific video to invalidate (all if not specified)")
    invalidate_parser.add_argument("--all", action="store_true", help="Invalidate all cache entries")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize cache manager
    try:
        cache_manager = VideoCacheManager(args.config)
    except Exception as e:
        print(f"❌ Failed to initialize cache manager: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == "stats":
            show_cache_stats(cache_manager)
        
        elif args.command == "list":
            list_cached_videos(cache_manager)
        
        elif args.command == "cleanup":
            cleanup_cache(cache_manager)
        
        elif args.command == "invalidate":
            if args.all or not args.video:
                invalidate_cache(cache_manager)
            else:
                invalidate_cache(cache_manager, args.video)
        
        else:
            print(f"Unknown command: {args.command}")
            
    except Exception as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()