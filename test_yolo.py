from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/lp_model3/weights/best.pt")

img_path = "test.jpg"  # use any sample image with vehicles
results = model(img_path)  # removed show=True

for r in results:
    if hasattr(r, 'boxes') and r.boxes is not None:
        print("Detected:", len(r.boxes), "objects")
        print("Class IDs:", r.boxes.cls)
        print("Confidences:", r.boxes.conf)
        print("Classes:", [model.names[int(c)] for c in r.boxes.cls])
    else:
        print("No boxes detected.")