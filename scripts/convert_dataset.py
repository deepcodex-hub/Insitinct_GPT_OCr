import json
import os
from pathlib import Path
from ultralytics.data.converter import convert_coco

dataset_dir = Path(r"D:\GPT_instinct\dataset")

# 1. Print classes from COCO JSON to build YAML
coco_json_path = dataset_dir / "train" / "_annotations.coco.json"
if coco_json_path.exists():
    print(f"Found COCO annotations at {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    print("Categories:", [c['name'] for c in categories])
    
    # 2. Convert COCO to YOLO
    print("Converting COCO to YOLO format...")
    convert_coco(
        labels_dir=str(dataset_dir / "train"),
        use_segments=False,
        use_keypoints=False,
        cls91to80=False
    )
    print("Conversion finished.")
else:
    print(f"COCO annotations NOT found at {coco_json_path}")
