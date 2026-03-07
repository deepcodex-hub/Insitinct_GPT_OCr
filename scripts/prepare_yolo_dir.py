import os
import shutil
from pathlib import Path

src_images = Path(r"D:\GPT_instinct\dataset\train")
src_labels = Path(r"D:\GPT_instinct\coco_converted2\labels\_annotations.coco")

dest_images = Path(r"D:\GPT_instinct\yolo_dataset\images\train")
dest_labels = Path(r"D:\GPT_instinct\yolo_dataset\labels\train")

dest_images.mkdir(parents=True, exist_ok=True)
dest_labels.mkdir(parents=True, exist_ok=True)

print("Copying images...")
for img in src_images.glob("*.jpg"):
    shutil.copy(img, dest_images / img.name)

print("Copying labels...")
for lbl in src_labels.glob("*.txt"):
    shutil.copy(lbl, dest_labels / lbl.name)

print("Dataset organized.")
