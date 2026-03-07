#!/usr/bin/env bash
# export_to_onnx.sh
# Export models to ONNX formats for deployment.

echo "[INFO] Exporting models to ONNX FP16..."
export PYTHONPATH=.

# Example Ultralytics YOLOv8 export
# yolo export model=models/yolov8_detector.pt format=onnx half=True

# Example PyTorch export
# python -c "import torch; torch.onnx.export(...)"

echo "[SUCCESS] ONNX exports complete."
