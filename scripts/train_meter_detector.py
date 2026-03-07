from ultralytics import YOLO

if __name__ == '__main__':
    print("Initializing YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")
    
    print("Starting training on meter_dataset...")
    results = model.train(
        data=r"D:\GPT_instinct\data\meter_dataset.yaml",
        epochs=1,
        imgsz=640,
        batch=16,
        name="meter_detector",
        device="cpu"  # Force CPU in case standard CUDA isn't available to prevent fast-crashing
    )
    
    print("Exporting best weights to ONNX format...")
    success = model.export(format="onnx", dynamic=True)
    print(f"Export successful: {success}")
