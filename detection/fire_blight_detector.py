"""
Fire Blight Detector using YOLOv8

This class handles loading a YOLOv8 model and running inference for fire blight detection.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class FireBlightDetector:
    """
    Fire Blight detector using YOLOv8 for disease detection in apple trees.
    
    This class loads a pre-trained YOLOv8 model and provides methods for running
    inference on images to detect fire blight symptoms.
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize the Fire Blight detector.
        
        Args:
            model_path: Path to the YOLOv8 model weights (.pt file).
                       If None, looks for 'best.pt' in detection/models/ directory.
            confidence_threshold: Minimum confidence score for detections (0-1).
        """
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Please install it with:\n"
                "  pip install ultralytics\n"
                "or\n"
                "  conda install -c conda-forge ultralytics"
            )
        
        # Determine model path
        if model_path is None:
            # Default to best.pt in the models directory
            script_dir = Path(__file__).parent
            model_path = script_dir / "models" / "best.pt"
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                f"Please place your YOLOv8 model weights (.pt file) in:\n"
                f"  {script_dir / 'models' / 'best.pt'}\n"
                f"or provide a custom path when initializing the detector."
            )
        
        print(f"Loading YOLOv8 model from: {model_path}")
        self.model = self.YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold
        print(f"Model loaded successfully. Confidence threshold: {confidence_threshold}")
        
        # Cache class names
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        print(f"Detected classes: {self.class_names}")
    
    def detect(self, image: np.ndarray, visualize: bool = False) -> List[Dict]:
        """
        Run fire blight detection on an image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB or BGR format
            visualize: If True, return annotated image
            
        Returns:
            List of detection dictionaries, each containing:
                - 'bbox': [x1, y1, x2, y2] bounding box coordinates
                - 'confidence': confidence score (0-1)
                - 'class_id': integer class ID
                - 'class_name': string class name
                - 'center': [cx, cy] center of bounding box
        """
        # Run inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            # Extract detections
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                
                # Calculate center
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'center': [float(cx), float(cy)]
                }
                
                detections.append(detection)
        
        if visualize:
            # Get annotated image
            annotated_img = results[0].plot() if len(results) > 0 else image
            return detections, annotated_img
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Run detection on a batch of images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of detection lists (one per image)
        """
        # Run batch inference
        results = self.model(images, conf=self.confidence_threshold, verbose=False)
        
        batch_detections = []
        
        for result in results:
            detections = []
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'center': [float(cx), float(cy)]
                    }
                    
                    detections.append(detection)
            
            batch_detections.append(detections)
        
        return batch_detections
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """
        Get a summary of detections.
        
        Args:
            detections: List of detection dictionaries from detect()
            
        Returns:
            Dictionary with summary statistics:
                - 'total_detections': total number of detections
                - 'class_counts': dictionary mapping class names to counts
                - 'avg_confidence': average confidence across all detections
                - 'max_confidence': maximum confidence score
        """
        if not detections:
            return {
                'total_detections': 0,
                'class_counts': {},
                'avg_confidence': 0.0,
                'max_confidence': 0.0
            }
        
        class_counts = {}
        confidences = []
        
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(det['confidence'])
        
        return {
            'total_detections': len(detections),
            'class_counts': class_counts,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'max_confidence': np.max(confidences) if confidences else 0.0
        }
    
    def set_confidence_threshold(self, threshold: float):
        """
        Update the confidence threshold.
        
        Args:
            threshold: New confidence threshold (0-1)
        """
        self.confidence_threshold = np.clip(threshold, 0.0, 1.0)
        print(f"Confidence threshold updated to: {self.confidence_threshold}")


def demo():
    """Demo function showing how to use the FireBlightDetector."""
    import matplotlib.pyplot as plt
    
    print("Fire Blight Detector Demo")
    print("=" * 60)
    
    try:
        # Initialize detector
        detector = FireBlightDetector(confidence_threshold=0.5)
        
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        print("\nRunning detection on dummy image...")
        detections = detector.detect(dummy_image)
        
        print(f"\nDetections found: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"  Detection {i+1}:")
            print(f"    Class: {det['class_name']}")
            print(f"    Confidence: {det['confidence']:.3f}")
            print(f"    BBox: {det['bbox']}")
        
        # Get summary
        summary = detector.get_detection_summary(detections)
        print(f"\nSummary:")
        print(f"  Total detections: {summary['total_detections']}")
        print(f"  Class counts: {summary['class_counts']}")
        print(f"  Avg confidence: {summary['avg_confidence']:.3f}")
        
        print("\n" + "=" * 60)
        print("Demo complete!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to:")
        print("  1. Install ultralytics: pip install ultralytics")
        print("  2. Place your model file in: detection/models/best.pt")


if __name__ == "__main__":
    demo()
