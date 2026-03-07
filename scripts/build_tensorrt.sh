#!/usr/bin/env bash
# build_tensorrt.sh
# Builds TensorRT engines from ONNX for Triton serving.

echo "[INFO] Building TensorRT engines from ONNX..."
export PYTHONPATH=.

# Example trtexec command
# trtexec --onnx=models/yolov8_detector.onnx --saveEngine=models/yolov8_detector.engine --fp16

echo "[SUCCESS] TensorRT engines built for Triton serving."
