import os
import shutil
import cv2
import json
import glob
from pathlib import Path

# Need to import the pipeline logic to generate the debug artifacts
# Since FastAPI app runs it via POST /infer, it's easier to just run the raw functions or start the server and curl it.
# Or just copy existing debug artifacts to examples/debug_session/
# Wait, the artifacts are already generated in `debug_artifacts/` by `app.py` when it processes images.
# Let's run `app.py` logic directly.

from api.app import detector, dewarper, sr_wrapper, decimal_detector, trocr, paddle, easyocr, rover, validator, llm, calibrator
from ag_module.image_quality import analyze_image_quality
from ag_module.expand_and_color_fallback import expand_bbox, get_color_fallback_crop

def process_and_save_debug(img_path, out_dir, img_id):
    os.makedirs(out_dir, exist_ok=True)
    
    image = cv2.imread(img_path)
    if image is None: return

    # Save original
    cv2.imwrite(f"{out_dir}/{img_id}_01_original.jpg", image)

    detections = detector.detect(image)
    img_h, img_w = image.shape[:2]
    
    target_field = None
    color_mask = None
    expanded_bbox_img = None
    
    if detections:
        disp_dets = [d for d in detections if d['class'] == 'display']
        best_det = max(disp_dets, key=lambda x: x['confidence']) if disp_dets else detections[0]
        nx1, ny1, nx2, ny2 = expand_bbox(best_det['bbox'], img_w, img_h, scale=0.15)
        target_field = image[ny1:ny2, nx1:nx2]
        expanded_bbox_img = image.copy()
        cv2.rectangle(expanded_bbox_img, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)

    if target_field is None or (target_field.shape[0]*target_field.shape[1] < img_w*img_h*0.02):
        crop_fb, c_mask, bbox_fb = get_color_fallback_crop(image)
        if crop_fb is not None:
            target_field = crop_fb
            color_mask = c_mask
            
    if target_field is None: target_field = image

    cv2.imwrite(f"{out_dir}/{img_id}_02_expanded_crop.jpg", target_field)

    warped = dewarper.apply_dewarp(target_field)
    
    try:
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        warped = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except:
        pass

    h, w = warped.shape[:2]
    if w < 300:
        enhanced = sr_wrapper.enhance(warped)
    else:
        enhanced = cv2.resize(warped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    MAX_OCR_WIDTH = 1024
    h, w = enhanced.shape[:2]
    if w > MAX_OCR_WIDTH:
        new_h = int(h * (MAX_OCR_WIDTH / w))
        enhanced = cv2.resize(enhanced, (MAX_OCR_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(f"{out_dir}/{img_id}_03_sr_enhanced.jpg", enhanced)

    dec_conf = decimal_detector.detect(enhanced)
    trocr_res = trocr.recognize(enhanced)
    paddle_res = paddle.recognize(enhanced)
    easy_res = easyocr.recognize(enhanced)
    
    roved_res = rover.align_and_vote([trocr_res, paddle_res, easy_res])
    validated_res = validator.validate_and_correct(roved_res["text"], roved_res["confidence"], dec_conf, initial_candidates=roved_res.get("candidates", []))
    
    final_text = validated_res["value"]
    
    with open(f"{out_dir}/{img_id}_04_final_value.txt", "w") as f:
        f.write(f"Recognized Value: {final_text}\nConfidence: {validated_res['probability']:.4f}\nSources Voted: {json.dumps(roved_res.get('candidates'))}")

if __name__ == "__main__":
    benchmark_dir = "benchmark/100"
    out_dir = "examples/debug_session"
    os.makedirs(out_dir, exist_ok=True)
    images = glob.glob(f"{benchmark_dir}/*.jpg")[:5]
    if not images:
        images = glob.glob("dataset/train/*.jpg")[:5]
    for i, img_path in enumerate(images):
        print(f"Processing debug session pair {i+1}/5: {img_path}")
        process_and_save_debug(img_path, out_dir, f"sample_{i+1}")
    print(f"Debug session generated in {out_dir}")
