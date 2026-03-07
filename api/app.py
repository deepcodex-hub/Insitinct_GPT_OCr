import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
import time

from ag_module.detector import MeterDetector # the old fallback logic
from detector.yolov8_adapter import YOLOv8Adapter # The new strict yolo logic
from ag_module.dewarp import DewarpProcessor
from ag_module.sr import RealESRGANWrapper
from ag_module.image_quality import analyze_image_quality
from ag_module.decimal_detector import DecimalDetectorConfig
from ag_module.expand_and_color_fallback import expand_bbox, get_color_fallback_crop
import uuid

if not os.path.exists("debug_artifacts"):
    os.makedirs("debug_artifacts")

from ocr_pipeline.trocr_adapter import TrOCRAdapter
from ocr_pipeline.paddle_adapter import PaddleAdapter
from ocr_pipeline.easyocr_adapter import EasyOCRAdapter
from ocr_pipeline.ensemble_rover import DecimalAwareRover
from ocr_pipeline.decimal_validator import DecimalValidator
from ocr_pipeline.calibrator import ModelCalibrator
from ocr_pipeline.llm_corrector import LLMCorrector

from api.schemas import OCRResponseSchema, FieldOutput, ArtifactURIs
from qc.labelstudio_hooks import push_to_qc

app = FastAPI(title="Production Anti-Gravity OCR API")

# Initialize models
try:
    detector = YOLOv8Adapter(model_path=r'D:\GPT_instinct\models\yolov8_detector.pt') # strict yolov8
except:
    detector = MeterDetector() # fallback

dewarper = DewarpProcessor()
sr_wrapper = RealESRGANWrapper(fp16=True)

# Detectors
decimal_detector = DecimalDetectorConfig(model_path=r'D:\GPT_instinct\models\weights\decimal_cnn_best.pt')

# OCR Engines
trocr = TrOCRAdapter()
paddle = PaddleAdapter()
easyocr = EasyOCRAdapter()

# Voting & Validation
rover = DecimalAwareRover(decimal_penalty=2.0)
validator = DecimalValidator(field_type="kwh") # Default to kWh for now
calibrator = ModelCalibrator() # Uses Isotonic Regression if fitted
llm = LLMCorrector()

def get_presigned_url(obj_path, expires_in=3600):
    """Stub: return an S3/MinIO presigned URL for debug artifacts."""
    return f"https://s3.agm-infra.internal/{obj_path}?sig=stub&exp={expires_in}"

@app.post("/infer", response_model=OCRResponseSchema)
async def infer(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    start_time = time.time()
    
    # 1. Read
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 2. Image Quality Analyzer
    print("Step 2: Analyzing image quality...")
    iq_flags = analyze_image_quality(image)
    if iq_flags["not_legible"]:
        # Return explicit not legible status if it's completely unreadable
        return OCRResponseSchema(
            image_id=file.filename,
            meter_serial=FieldOutput(value="N/A", probability=0.0),
            kwh=FieldOutput(value="N/A", probability=0.0),
            kvah=FieldOutput(value="N/A", probability=0.0),
            md_kw=FieldOutput(value="N/A", probability=0.0),
            demand_kva=FieldOutput(value="N/A", probability=0.0),
            image_quality=iq_flags,
            qc_flag=True,
            processing_latency_ms=int((time.time() - start_time) * 1000),
            artifacts=ArtifactURIs()
        )

    # 3. Detect & Crop
    print("Step 3: Running detection...")
    detections = detector.detect(image)
    img_h, img_w = image.shape[:2]
    
    target_field = None
    color_mask = None
    expanded_bbox_img = None
    
    if detections:
        # Get the highest confidence display detection or fallback to first
        disp_dets = [d for d in detections if d['class'] == 'display']
        best_det = max(disp_dets, key=lambda x: x['confidence']) if disp_dets else detections[0]
        
        # Expand bbox
        nx1, ny1, nx2, ny2 = expand_bbox(best_det['bbox'], img_w, img_h, scale=0.15)
        target_field = image[ny1:ny2, nx1:nx2]
        
        # Draw expanded bbox for debug
        expanded_bbox_img = image.copy()
        cv2.rectangle(expanded_bbox_img, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)

    # If YOLO fails or crop area is very small, use color fallback
    if target_field is None or (target_field.shape[0]*target_field.shape[1] < img_w*img_h*0.02):
        print("YOLO Box too small or missing. Running Color Fallback...")
        crop_fb, c_mask, bbox_fb = get_color_fallback_crop(image)
        if crop_fb is not None:
            target_field = crop_fb
            color_mask = c_mask
            print("Successfully applied color fallback.")
            
    if target_field is None:
        target_field = image # Absolute fallback

    # 4. Anti-Gravity Preprocessing
    warped = dewarper.apply_dewarp(target_field)
    
    # Apply CLAHE
    try:
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        warped = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"CLAHE failed: {e}")

    # SR Gating
    h, w = warped.shape[:2]
    if w < 300:
        print("Crop width < 300px, running Real-ESRGAN...")
        enhanced = sr_wrapper.enhance(warped)
    else:
        print("Crop width >= 300px, using Bicubic resize to avoid hallucination...")
        enhanced = cv2.resize(warped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Resize to a reasonable max width for OCR to avoid OOM
    MAX_OCR_WIDTH = 1024
    h, w = enhanced.shape[:2]
    if w > MAX_OCR_WIDTH:
        new_h = int(h * (MAX_OCR_WIDTH / w))
        enhanced = cv2.resize(enhanced, (MAX_OCR_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image for OCR from {w} to {MAX_OCR_WIDTH}")

    # 5. Decimal Pipeline (Score patch)
    print("Step 5: Running decimal detector...")
    dec_conf = decimal_detector.detect(enhanced)

    # 6. OCR Ensemble
    print("Step 6: Running OCR ensemble (TrOCR)...")
    trocr_res = trocr.recognize(enhanced)
    print("Step 6.1: Running PaddleOCR...")
    paddle_res = paddle.recognize(enhanced)
    print("Step 6.2: Running EasyOCR...")
    easy_res = easyocr.recognize(enhanced)

    # 7. ROVER Token Voting
    roved_res = rover.align_and_vote([trocr_res, paddle_res, easy_res])
    raw_text = roved_res["text"]
    raw_conf = roved_res["confidence"]

    # 8. Decimal Validator
    validated_res = validator.validate_and_correct(raw_text, raw_conf, dec_conf, initial_candidates=roved_res.get("candidates", []))
    
    # Adjust probability using Decimal Detector
    valid_prob = validated_res["probability"]
    boosted_prob = valid_prob * (0.5 + 0.5 * dec_conf)
    validated_res["probability"] = boosted_prob

    # 9. LLM Post-Processor
    llm_res = llm.correct(validated_res["value"])
    final_text = llm_res.get("best", validated_res["value"])

    # 10. Calibration (Isotonic Temp Scaling)
    calibrated_conf = calibrator.calibrate(validated_res["probability"])

    # Build Field
    kwh_field = FieldOutput(
        value=final_text,
        probability=calibrated_conf,
        sources=["trocr", "paddleocr", "easyocr"],
        decimals=validated_res["decimals"],
        candidates=validated_res["candidates"],
        debug={"raw_ocr": raw_text, "decimal_detector_score": dec_conf}
    )

    # 11. QC Trigger
    reason_codes = []
    if calibrated_conf < 0.98:
        reason_codes.append("LOW_CONFIDENCE")
    if len(validated_res["candidates"]) > 1:
        reason_codes.append("MULTIPLE_CONFORMAL_CANDIDATES")
        
    reject = len(reason_codes) > 0

    # Save debug artifacts locally
    uid = str(uuid.uuid4())[:8]
    base_name = f"debug_{uid}_{file.filename}"
    if target_field is not None: cv2.imwrite(f"debug_artifacts/crop_{base_name}.png", target_field)
    if expanded_bbox_img is not None: cv2.imwrite(f"debug_artifacts/expanded_{base_name}.png", expanded_bbox_img)
    if color_mask is not None: cv2.imwrite(f"debug_artifacts/colormask_{base_name}.png", color_mask)
    if enhanced is not None: cv2.imwrite(f"debug_artifacts/sr_{base_name}.png", enhanced)

    response = OCRResponseSchema(
        image_id=file.filename,
        meter_serial=FieldOutput(value="12345678", probability=0.99), # Mocked for demo
        kwh=kwh_field,
        kvah=FieldOutput(value="0.0", probability=0.0), # Mocked
        md_kw=FieldOutput(value="0.0", probability=0.0), # Mocked
        demand_kva=FieldOutput(value="0.0", probability=0.0), # Mocked
        image_quality=iq_flags,
        reason_codes=reason_codes,
        qc_flag=reject,
        processing_latency_ms=int((time.time() - start_time) * 1000),
        artifacts=ArtifactURIs(
            crop_url=get_presigned_url(f"crops/crop_{base_name}.png"),
            color_mask_url=get_presigned_url(f"crops/colormask_{base_name}.png") if color_mask is not None else None,
            sr_url=get_presigned_url(f"crops/sr_{base_name}.png"),
            alignment_map=get_presigned_url(f"alignment/{file.filename}")
        )
    )

    if reject:
        background_tasks.add_task(push_to_qc, file.filename, response.dict())

    return response

@app.get("/health")
def health():
    return {"status": "healthy", "version": "production_agm_ocr_v1"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
