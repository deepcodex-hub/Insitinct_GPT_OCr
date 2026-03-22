import os
import cv2
from ultralytics import YOLO

def diagnose():
    print("--- Streamlit Cloud Diagnostics ---")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Files in root: {os.listdir('.')}")
    
    model_path = "runs/detect/meter_detector4/weights/best.pt"
    print(f"Checking model path: {model_path}")
    print(f"Exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        try:
            model = YOLO(model_path)
            print("Model loaded successfully.")
            print(f"Model Names: {model.names}")
        except Exception as e:
            print(f"Model load error: {e}")
    else:
        print("CRITICAL: Model weights not found in the expected location!")

if __name__ == "__main__":
    diagnose()
