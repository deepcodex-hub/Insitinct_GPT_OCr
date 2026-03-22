import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")

image_path = r"examples\debug_session\single_test\test_screenshot_01_original.jpg"

img = cv2.imread(image_path)

results = model(img)

detections = []

for r in results:
    for box in r.boxes:

        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if conf < 0.25:
            continue

        x1,y1,x2,y2 = map(int,box.xyxy[0])

        center_x = (x1+x2)//2

        detections.append((center_x, cls, conf))

# sort digits left → right
detections.sort(key=lambda x: x[0])

reading = ""

print("\nDetected objects:\n")

for d in detections:

    digit_class = d[1]

    if digit_class == 10:
        digit = "."
    else:
        digit = str(digit_class)

    reading += digit

    print("Digit:",digit,"Confidence:",round(d[2],3))

# decimal correction
if "." not in reading and len(reading) >= 6:
    reading = reading[:-1] + "." + reading[-1]

print("\nFinal Meter Reading:",reading)