"""
Fix missing post-decimal digit annotations in YOLO label files.
Uses the trained YOLO model to detect digits in each image with aggressive parameters.
"""
import os
import glob
import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/meter_detector/weights/best.pt')

# Class mapping: 0-9 = digits, 10 = decimal
CLASS_NAMES = {i: str(i) for i in range(10)}
CLASS_NAMES[10] = '.'

def get_reading_from_labels(label_path):
    annotations = []
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append({
                    'class': cls_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                })
    annotations.sort(key=lambda a: a['x_center'])
    return annotations

def detect_missing_digits(image_path, decimal_x_center):
    """Run YOLO with aggressive settings to find digits to the right of the decimal."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    
    # Large padding to give context to edge digits
    pad_x = 100
    pad_y = 50
    padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    # Aggressive high-res inference
    results = model(padded, imgsz=1024, conf=0.01, iou=0.5, verbose=False)[0]
    
    potential_digits = []
    # Identify the boundary where the decimal point was (in normalized coordinates)
    decimal_right_edge = decimal_x_center + 0.02 # small buffer
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id >= 10: continue # ignore more decimals
        
        # Bbox in raw image coords
        x1 = float(box.xyxy[0][0]) - pad_x
        x2 = float(box.xyxy[0][2]) - pad_x
        y1 = float(box.xyxy[0][1]) - pad_y
        y2 = float(box.xyxy[0][3]) - pad_y
        
        # Normalized coords
        nx_center = ((x1 + x2) / 2) / w
        ny_center = ((y1 + y2) / 2) / h
        nw = (x2 - x1) / w
        nh = (y2 - y1) / h
        
        # Only keep if it's clearly to the right of the existing decimal
        if nx_center > decimal_right_edge:
            potential_digits.append({
                'class': cls_id,
                'x_center': nx_center,
                'y_center': ny_center,
                'width': nw,
                'height': nh,
                'conf': float(box.conf[0])
            })
    
    # Sort by confidence and then by x_center to get the most likely next digit
    # If there are multiple, keep the one closest to the decimal x_center (first digit after decimal)
    potential_digits.sort(key=lambda x: x['x_center'])
    
    # Only return the next 1-2 digits to avoid picking up far-right noise
    if potential_digits:
        return [potential_digits[0]] # Just return the first one after the decimal for stability
    return []

def fix_all():
    base_dir = r'c:\Users\Deephika\Insitinct_GPT_OCr\dataset'
    total_added = 0
    
    for split in ['train', 'valid', 'test']:
        labels = glob.glob(os.path.join(base_dir, split, 'labels', '*.txt'))
        print(f"Split {split}: {len(labels)} labels")
        
        for lf in labels:
            anns = get_reading_from_labels(lf)
            if not anns: continue
            
            # Find the rightmost decimal
            decimal_ann = None
            for a in reversed(anns):
                if a['class'] == 10:
                    decimal_ann = a
                    break
            
            if not decimal_ann: continue
            
            # Check if any digit already exists after it
            has_post = any(a['x_center'] > decimal_ann['x_center'] and a['class'] != 10 for a in anns)
            if has_post: continue
            
            # Missing post-decimal digit!
            img_path = lf.replace('\\labels\\', '\\images\\').replace('.txt', '.jpg')
            if not os.path.exists(img_path):
                img_path = img_path.replace('.jpg', '.png')
                if not os.path.exists(img_path): continue
            
            new_digits = detect_missing_digits(img_path, decimal_ann['x_center'])
            if new_digits:
                with open(lf, 'a') as f:
                    for d in new_digits:
                        f.write(f"\n{d['class']} {d['x_center']} {d['y_center']} {d['width']} {d['height']}")
                total_added += 1
                reading = "".join([CLASS_NAMES.get(a['class'], '?') for a in anns])
                added = "".join([str(d['class']) for d in new_digits])
                print(f"  [{os.path.basename(lf)}] {reading} -> {reading}{added} (conf {new_digits[0]['conf']:.2f})")

    print(f"\nSUCCESS: Added missing digits to {total_added} label files.")

if __name__ == '__main__':
    fix_all()
