#!/bin/bash
# train_yolo.sh
# Script to train YOLOv8 on the 1500 image dataset

set -e

DATASET_YAML="data/meter_dataset.yaml"
EPOCHS=100
BATCH_SIZE=16
IMG_SIZE=640
MODEL="yolov8n.pt" # Start from pretrained nano model

echo "Starting YOLOv8 training on $DATASET_YAML..."
yolo task=detect mode=train model=$MODEL data=$DATASET_YAML epochs=$EPOCHS imgsz=$IMG_SIZE batch=$BATCH_SIZE device=0 name=meter_detector

echo "Exporting best model to ONNX..."
yolo export model=runs/detect/meter_detector/weights/best.pt format=onnx dynamic=True

echo "Training complete. Best weights saved to runs/detect/meter_detector/weights/best.pt"
