import argparse
import cv2
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ag_module.detector import MeterDetector
from ag_module.dewarp import DewarpProcessor
from ag_module.sr import RealESRGANWrapper
from ocr_pipeline.recognizer import OCRRecognizer
from ocr_pipeline.llm_corrector import LLMCorrector

def execute_inference(image_path, output_json=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Initialize components
    detector = MeterDetector()
    dewarper = DewarpProcessor()
    sr_wrapper = RealESRGANWrapper(fp16=True)
    recognizer = OCRRecognizer(use_gpu=False)
    llm_corrector = LLMCorrector()

    # 1. Pipeline: Detect
    detections = detector.detect(image)
    if detections:
        target_field = detector.crop_detected_fields(image, detections)[0]
    else:
        # Fallback for pre-cropped images (like 1_cropped_158)
        print("No meter screen detected, assuming input image is the crop.")
        target_field = image

    # 2. AGM: Dewarp & SR
    warped = dewarper.apply_dewarp(target_field)
    enhanced = sr_wrapper.enhance(warped)
    
    cv2.imwrite(os.path.join("outputs", "debug_target.jpg"), target_field)
    cv2.imwrite(os.path.join("outputs", "debug_warped.jpg"), warped)
    cv2.imwrite(os.path.join("outputs", "debug_enhanced.jpg"), enhanced)
    
    # 3. Custom YOLO Digit OCR (replaces TrOCR/Paddle)
    from ultralytics import YOLO
    try:
        model_path = r"runs/detect/meter_detector4/weights/best.pt"
        if not os.path.exists(model_path):
            model_path = r"runs/detect/meter_detector/weights/best.pt"
        print(f"Loading digit model from: {model_path}")
        digit_model = YOLO(model_path)
        print(f"Model Classes: {digit_model.names}")
        
        # Use the RAW target field for detection (SR/Dewarp can sometimes distort sharp edges of digital digits)
        pad = 100
        padded_img = cv2.copyMakeBorder(target_field, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # IMPROVE CONTRAST (Helpful for thin digits like '1')
        gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)
        # Background compensation (useful for reflecting meter screens)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        final_img = cv2.add(gray, blackhat)
        
        # run on padded raw image with optimized threshold
        print(f"Running YOLO inference on image shape: {padded_img.shape}")
        results = digit_model(final_img, imgsz=1280, conf=0.08, iou=0.25)[0] 
        
        # 1. Deduplicate boxes (extra NMS layer for robustness at low conf)
        # Use simple IOU check
        raw_digits = []
        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy()
            raw_digits.append({
                "bbox": b,
                "cls": int(box.cls[0]),
                "conf": float(box.conf[0])
            })
        
        # Sort by confidence and remove overlaps
        final_boxes = []
        raw_digits.sort(key=lambda x: x['conf'], reverse=True)
        for d in raw_digits:
            keep = True
            for f in final_boxes:
                # Basic overlap check
                ix1 = max(d['bbox'][0], f['bbox'][0])
                iy1 = max(d['bbox'][1], f['bbox'][1])
                ix2 = min(d['bbox'][2], f['bbox'][2])
                iy2 = min(d['bbox'][3], f['bbox'][3])
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                area_d = (d['bbox'][2]-d['bbox'][0]) * (d['bbox'][3]-d['bbox'][1])
                if inter / area_d > 0.5:
                    keep = False
                    break
            if keep:
                final_boxes.append(d)

        digits = []
        for d in final_boxes:
            x1 = d['bbox'][0] - pad
            cls_id = d['cls']
            conf = d['conf']
            # Classes: 0='0', ..., 9='9', 10='d'
            cls_name = str(cls_id) if cls_id < 10 else '.'
            digits.append((x1, cls_name, conf))
        
        print(f"Detected {len(digits)} digits.")
        
        # DRAW BOXES FOR DEBUG
        debug_img = padded_img.copy()
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            bcl = int(box.cls[0])
            bcn = str(bcl) if bcl < 10 else '.'
            bcf = float(box.conf[0])
            cv2.rectangle(debug_img, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)
            cv2.putText(debug_img, f"{bcn} {bcf:.2f}", (int(bx1), int(by1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(os.path.join("outputs", "debug_target.jpg"), debug_img)
        
        # Sort left to right
        digits.sort(key=lambda x: x[0])
        raw_text = "".join([d[1] for d in digits])
        avg_conf = sum([d[2] for d in digits]) / len(digits) if digits else 0.0
        
        roved_res = {
            "text": raw_text if raw_text else "0",
            "confidence": avg_conf
        }
    except Exception as e:
        print(f"Custom YOLO failed: {e}")
        raw_text = "0"
        roved_res = {"text": "0", "confidence": 0.0}
    
    # 4. LLM Correction
    corrected_res = llm_corrector.correct(raw_text)
    
    result = {
        "raw_text": str(raw_text),
        "corrected_text": str(corrected_res.get('best', raw_text)),
        "confidence": float(roved_res['confidence']),
        "reject_to_qc": bool(roved_res['confidence'] < 0.85),
        "detections": [
            {
                "bbox": [int(v) for v in det['bbox']],
                "class": int(det['cls']),
                "confidence": float(det['conf'])
            } for det in final_boxes
        ]
    }

    print(json.dumps(result, indent=4))
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()
    
    execute_inference(args.image, args.output)
