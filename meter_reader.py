from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")

image_path = "examples/debug_session/single_test/test_screenshot_01_original.jpg"

results = model(image_path)

detections = []

print("\nDetected objects:\n")

for r in results:
    for box in r.boxes:

        conf = float(box.conf[0])
        cls = int(box.cls[0])

        x1 = float(box.xyxy[0][0])
        x2 = float(box.xyxy[0][2])

        center_x = (x1 + x2) / 2

        print("Class:", cls, "Confidence:", conf)

        if conf < 0.25:
            continue

        detections.append((center_x, cls))

# sort left → right
detections = sorted(detections, key=lambda d: d[0])

reading = ""

for _, cls in detections:

    if cls == 10:
        reading += "."
    else:
        reading += str(cls)

print("\nFinal Meter Reading:", reading)