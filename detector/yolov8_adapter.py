import cv2
import numpy as np

class YOLOv8Adapter:
    """Wrapper for the YOLOv8 detector using the local dataset."""
    def __init__(self, model_path='yolov8n.pt'):
        self.model_path = model_path
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.mock_mode = False
        except (ImportError, Exception):
            print(f"Warning: Could not load YOLOv8 model at {model_path}. Running adapter in mock mode.")
            self.model = None
            self.mock_mode = True

    def detect(self, image: np.ndarray, conf=0.5):
        """Detects meter display and serial number fields in the image."""
        if self.mock_mode:
            # Fallback mock logic for the structural pipeline:
            h, w = image.shape[:2]
            return [
                {"bbox": [int(w*0.2), int(h*0.3), int(w*0.8), int(h*0.5)], "class": "display", "confidence": 0.85},
                {"bbox": [int(w*0.3), int(h*0.7), int(w*0.7), int(h*0.85)], "class": "serial", "confidence": 0.90}
            ]
        
        results = self.model(image, conf=conf)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_idx = int(box.cls[0].cpu().numpy())
                # Assume 0 is display, 1 is serial in our custom trained model
                cls_name = "display" if cls_idx == 0 else "serial" 
                conf_score = float(box.conf[0].cpu().numpy())
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class": cls_name,
                    "confidence": conf_score
                })
        return detections

    def crop_detected_fields(self, image: np.ndarray, detections: list):
        """Returns cropped numpy arrays for each detection dict."""
        crops = {}
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            crop = image[y1:y2, x1:x2]
            crops[det['class']] = crop
        return crops
