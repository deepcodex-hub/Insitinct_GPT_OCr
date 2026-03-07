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

def run_infer(image_path, output_json=None):
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
        target_field = image

    # 2. AGM: Dewarp & SR
    warped = dewarper.apply_dewarp(target_field)
    enhanced = sr_wrapper.enhance(warped)
    
    cv2.imwrite(os.path.join("outputs", "debug_target.jpg"), target_field)
    cv2.imwrite(os.path.join("outputs", "debug_warped.jpg"), warped)
    cv2.imwrite(os.path.join("outputs", "debug_enhanced.jpg"), enhanced)
    
    # 3. OCR (TrOCR, Paddle, Easy -> ROVER)
    trocr_res = recognizer.recognize_trocr(enhanced)
    paddle_res = recognizer.recognize_paddle(enhanced)
    easy_res = recognizer.recognize_easy(enhanced)
    
    roved_res = recognizer.ensemble_vote([trocr_res, paddle_res, easy_res])
    raw_text = roved_res['text']
    
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
                "class": int(det['class']),
                "confidence": float(det['confidence'])
            } for det in detections
        ] if isinstance(detections, list) else detections
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
    
    run_infer(args.image, args.output)
