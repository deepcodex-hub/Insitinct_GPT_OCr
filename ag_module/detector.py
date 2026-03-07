# from ultralytics import YOLO
import cv2
import numpy as np

class MeterDetector:
    def __init__(self, model_path='yolov8n.pt'):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.mock_mode = False
        except (ImportError, Exception):
            print("Waring: Could not load YOLOv8. Running MeterDetector in mock mode.")
            self.model = None
            self.mock_mode = True

    def detect(self, image: np.ndarray, conf=0.25):
        """Detects fields in the image."""
        if self.mock_mode:
            # Optimized fallback: Find the bright green LCD screen via HSV thresholding
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Define bounds for typical glowing green LCDs
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Morphological operations to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                # Get the largest green blob (the screen)
                c = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                
                # Add a small pad to not cut off digit edges
                pad_x = int(w * 0.05)
                pad_y = int(h * 0.1)
                
                img_h, img_w = image.shape[:2]
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img_w, x + w + pad_x)
                y2 = min(img_h, y + h + pad_y)
                
                if w > 30 and h > 15: # Sanity
                    return [{
                        "bbox": [x1, y1, x2, y2],
                        "class": 0,
                        "confidence": 0.85
                    }]
            
            # Absolute worst-case fallback (fail gracefully to center)
            h, w = image.shape[:2]
            return [{
                "bbox": [w//4, h//4, 3*w//4, 3*h//4],
                "class": 0,
                "confidence": 0.3
            }]
        
        results = self.model(image, conf=conf)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                conf_score = float(box.conf[0].cpu().numpy())
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class": cls,
                    "confidence": conf_score
                })
        return detections

    def crop_detected_fields(self, image: np.ndarray, detections: list):
        """Crops detected fields from the original image."""
        crops = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            crop = image[y1:y2, x1:x2]
            crops.append(crop)
        return crops
