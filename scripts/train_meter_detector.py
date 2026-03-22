from ultralytics import YOLO

if __name__ == '__main__':
    print("Initializing YOLOv8 Small model...")
    model = YOLO("yolov8s.pt")
    
    print("Starting training on meter_dataset...")
    results = model.train(
        data=r"c:\Users\Deephika\Insitinct_GPT_OCr\dataset\data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        name="meter_detector",
        device=0  # Using RTX 4050 GPU as requested
    )
    
    print("Exporting best weights to ONNX format...")
    success = model.export(format="onnx", dynamic=True)
    print(f"Export successful: {success}")
