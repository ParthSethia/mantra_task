import cv2
import numpy as np
import yaml
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random
import traceback

@dataclass
class Detection:
    """Single object detection"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str

@dataclass
class Track:
    """Object track with persistent ID"""
    track_id: int
    class_name: str
    detections: List[Dict]  # List of {timestamp, bbox, confidence}
    color: Tuple[int, int, int]  # RGB color for visualization
    first_seen: float
    last_seen: float
    active: bool = True

class ObjectTracker:
    """Object detection and tracking system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.tracking_config = config['object_tracking']
        self.cache_config = config['cache']
        
        # Initialize models
        self.detector = None
        self.tracker = None
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.colors = self._generate_colors(100)  # Pre-generate colors
        
        # YOLO class names (COCO dataset)
        self.yolo_classes = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
        
        if self.tracking_config['enabled']:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize YOLO and tracking models"""
        try:
            from ultralytics import YOLO
            from deep_sort_realtime.deepsort_tracker import DeepSort
            
            # Try multiple DeepSort import methods
            # DeepSort = None
            # import_attempts = [
            #     "from deep_sort_realtime import DeepSort",
            #     "from deep_sort_realtime.deepsort_tracker import DeepSort", 
            #     "from deep_sort_realtime.deep_sort_realtime import DeepSort",
            #     "from deep_sort_realtime.deep_sort import DeepSort"
            # ]
            
            # for attempt in import_attempts:
            #     try:
            #         exec(f"global DeepSort; {attempt}")
            #         if 'DeepSort' in locals() or 'DeepSort' in globals():
            #             break
            #     except ImportError:
            #         continue
            
            # if DeepSort is None:
            #     raise ImportError("Could not import DeepSort from any known path")
            
            # Initialize YOLO
            model_size = self.tracking_config['model_size']
            model_path = f"yolo11{model_size}.pt"
            print(f"Loading YOLO model: {model_path}")
            self.detector = YOLO(model_path)
            
            # Initialize DeepSORT
            print("Initializing DeepSORT tracker...")
            self.tracker = DeepSort(
                max_age=50,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.3,
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None
            )
            
            print("Object tracking models initialized successfully")
            
        except ImportError as e:
            print(f"Required packages not found: {e}")
            print("\nTrying alternative DeepSort installation...")
            print("Please try one of these installation methods:")
            print("1. pip install deep-sort-realtime==1.3.2")
            print("2. pip install git+https://github.com/levan92/deep_sort_realtime")
            print("3. Or disable object tracking in config.yaml: object_tracking.enabled = false")
            self.tracking_config['enabled'] = False
        except Exception as e:
            print(f"Failed to initialize tracking models: {e}")
            traceback.print_exc()
            print("Object tracking disabled. Video processing will continue without object tracking.")
            self.tracking_config['enabled'] = False
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for track visualization"""
        colors = []
        for i in range(num_colors):
            # Generate distinct colors using HSV
            hue = (i * 137.508) % 360  # Golden angle approximation
            saturation = 90
            value = 90
            
            # Convert HSV to RGB
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue/360, saturation/100, value/100)
            colors.append(tuple(int(c * 255) for c in rgb))
        
        return colors
    
    def detect_objects(self, frame: np.ndarray, timestamp: float = None) -> List[Detection]:
        """Detect objects in a frame"""
        if not self.tracking_config['enabled'] or self.detector is None:
            return []
        
        try:
            # Run YOLO detection
            results = self.detector(frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Get detection data
                        bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Filter by confidence and class
                        if (confidence >= self.tracking_config['confidence_threshold'] and
                            class_id < len(self.yolo_classes)):
                            
                            class_name = self.yolo_classes[class_id]
                            
                            # Filter by track_classes if specified
                            if (not self.tracking_config['track_classes'] or 
                                class_name in self.tracking_config['track_classes']):
                                
                                detections.append(Detection(
                                    bbox=tuple(bbox),
                                    confidence=confidence,
                                    class_id=class_id,
                                    class_name=class_name
                                ))
            
            return detections
            
        except Exception as e:
            print(f"Object detection failed: {e}")
            return []
    
    def update_tracks(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """Update object tracks with new detections"""
        if not self.tracking_config['enabled'] or self.tracker is None:
            return []
        
        try:
            # Prepare detections for DeepSORT
            raw_detections = []
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                raw_detections.append([x1, y1, x2-x1, y2-y1, det.confidence])
            
            # Update tracker
            tracks_output = self.tracker.update_tracks(raw_detections, frame=None)
            
            current_active_tracks = []
            
            for track in tracks_output:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                bbox = track.to_ltrb()  # Get bbox as [left, top, right, bottom]
                
                # Find corresponding detection for class info
                detection_class = "unknown"
                detection_confidence = 0.0
                for det in detections:
                    det_center = ((det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2)
                    track_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    
                    # Simple distance-based matching
                    distance = ((det_center[0] - track_center[0])**2 + (det_center[1] - track_center[1])**2)**0.5
                    if distance < 50:  # Threshold for matching
                        detection_class = det.class_name
                        detection_confidence = det.confidence
                        break
                
                # Create or update track
                if track_id not in self.tracks:
                    # New track
                    color = self.colors[track_id % len(self.colors)]
                    self.tracks[track_id] = Track(
                        track_id=track_id,
                        class_name=detection_class,
                        detections=[],
                        color=color,
                        first_seen=timestamp,
                        last_seen=timestamp,
                        active=True
                    )
                
                # Update track
                track_obj = self.tracks[track_id]
                track_obj.detections.append({
                    'timestamp': timestamp,
                    'bbox': [int(x) for x in bbox],
                    'confidence': detection_confidence,
                    'class_name': detection_class
                })
                track_obj.last_seen = timestamp
                track_obj.active = True
                
                current_active_tracks.append(track_obj)
            
            # Mark inactive tracks
            for track_id, track in self.tracks.items():
                if track.last_seen < timestamp - 1.0:  # 1 second threshold
                    track.active = False
            
            return current_active_tracks
            
        except Exception as e:
            print(f"Track update failed: {e}")
            return []
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[List[Track], np.ndarray]:
        """Process a frame: detect objects and update tracks"""
        if not self.tracking_config['enabled']:
            return [], frame
        
        # Detect objects
        detections = self.detect_objects(frame, timestamp)
        
        # Update tracks
        active_tracks = self.update_tracks(detections, timestamp)
        
        # Draw tracks on frame (optional visualization)
        annotated_frame = self._draw_tracks(frame.copy(), active_tracks)
        
        return active_tracks, annotated_frame
    
    def _draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw bounding boxes and track IDs on frame"""
        for track in tracks:
            if not track.detections:
                continue
            
            # Get latest detection
            latest_det = track.detections[-1]
            bbox = latest_det['bbox']
            confidence = latest_det['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), track.color, 2)
            
            # Draw label
            label = f"{track.class_name}_{track.track_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), track.color, -1)
            
            # Text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def get_track_summary(self) -> Dict:
        """Get summary of all tracks"""
        summary = {
            'total_tracks': len(self.tracks),
            'active_tracks': len([t for t in self.tracks.values() if t.active]),
            'track_classes': defaultdict(int),
            'tracks_detail': []
        }
        
        for track in self.tracks.values():
            summary['track_classes'][track.class_name] += 1
            
            # Calculate track duration
            duration = track.last_seen - track.first_seen
            
            summary['tracks_detail'].append({
                'track_id': track.track_id,
                'class_name': track.class_name,
                'first_seen': track.first_seen,
                'last_seen': track.last_seen,
                'duration': duration,
                'detection_count': len(track.detections),
                'active': track.active,
                'color': track.color
            })
        
        return summary
    
    def save_tracking_data(self, output_path: str):
        """Save tracking data to JSON file"""
        tracking_data = {
            'config': self.tracking_config,
            'summary': self.get_track_summary(),
            'tracks': {}
        }
        
        # Convert tracks to serializable format
        for track_id, track in self.tracks.items():
            tracking_data['tracks'][str(track_id)] = {
                'track_id': track.track_id,
                'class_name': track.class_name,
                'first_seen': track.first_seen,
                'last_seen': track.last_seen,
                'color': track.color,
                'active': track.active,
                'detections': track.detections
            }
        
        with open(output_path, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        print(f"Tracking data saved to: {output_path}")
    
    def load_tracking_data(self, input_path: str) -> bool:
        """Load tracking data from JSON file"""
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Restore tracks
            self.tracks = {}
            for track_id_str, track_data in data['tracks'].items():
                track_id = int(track_id_str)
                self.tracks[track_id] = Track(
                    track_id=track_data['track_id'],
                    class_name=track_data['class_name'],
                    detections=track_data['detections'],
                    color=tuple(track_data['color']),
                    first_seen=track_data['first_seen'],
                    last_seen=track_data['last_seen'],
                    active=track_data['active']
                )
            
            self.next_track_id = max(self.tracks.keys()) + 1 if self.tracks else 1
            
            print(f"Tracking data loaded from: {input_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load tracking data: {e}")
            return False


if __name__ == "__main__":
    # Test object tracking
    tracker = ObjectTracker()
    
    if not tracker.tracking_config['enabled']:
        print("Object tracking is disabled. Please install required packages:")
        print("pip install ultralytics deep-sort-realtime")
    else:
        print("Object tracker initialized successfully")
        print(f"Tracking classes: {tracker.tracking_config['track_classes']}")
        print(f"Confidence threshold: {tracker.tracking_config['confidence_threshold']}")
        print("Ready for video processing...")