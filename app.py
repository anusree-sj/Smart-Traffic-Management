import streamlit as st
import cv2
import tempfile
import time
import pytesseract
from ultralytics import YOLO

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="Smart Traffic Surveillance", layout="wide")
st.title("🚦 Smart Traffic Surveillance System with ANPR")

# -------------------------------
# Load YOLO Models
# -------------------------------
@st.cache_resource
def load_models():
    try:
        vehicle_model = YOLO("yolov8n.pt")  # Pre-trained YOLO for vehicles
        plate_model = YOLO("runs/detect/lp_model3/weights/best.pt")  # Trained LP model
        return vehicle_model, plate_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Settings")
enable_ocr = st.sidebar.checkbox("Enable License Plate Recognition (OCR)", value=True)
frame_skip = st.sidebar.slider("Frame Skip (for faster processing)", 1, 10, 3)

# -------------------------------
# Video Upload Section
# -------------------------------
uploaded_video = st.file_uploader("📹 Upload a traffic video", type=["mp4", "avi", "mov"])
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    st.video(video_path)
else:
    st.warning("Please upload a video to begin analysis.")
    st.stop()

# -------------------------------
# Process Video
# -------------------------------
if st.button("▶ Start Smart Analysis"):
    st.info("🧠 Analyzing traffic video... Please wait ⏳")

    vehicle_model, plate_model = load_models()
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = "annotated_output.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_counts = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0}
    detected_plates = set()  # Use set to store unique plates
    frame_counter = 0

    progress = st.progress(0)
    frame_display = st.image([])  # live frame placeholder

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        frame_boxes = set()  # store boxes in current frame to avoid double counting

        # -------------------------------
        # Vehicle Detection
        # -------------------------------
        vehicle_results = vehicle_model(frame)
        if vehicle_results:
            for r in vehicle_results:
                if r.boxes is None:
                    continue
                for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    if conf < 0.5:
                        continue
                    cls_name = vehicle_model.names[int(cls_id)]
                    if cls_name not in total_counts:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    # round box coordinates to reduce duplicates
                    box_tuple = (round(x1/10)*10, round(y1/10)*10, round(x2/10)*10, round(y2/10)*10)

                    if box_tuple not in frame_boxes:
                        total_counts[cls_name] += 1
                        frame_boxes.add(box_tuple)

                    # Draw vehicle box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, cls_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # -------------------------------
        # License Plate Detection
        # -------------------------------
        if enable_ocr:
            lp_results = plate_model(frame)
            if lp_results:
                for lp in lp_results:
                    if lp.boxes is None:
                        continue
                    for pb in lp.boxes.xyxy:
                        px1, py1, px2, py2 = map(int, pb)
                        # skip very tiny boxes
                        if (px2 - px1) < 20 or (py2 - py1) < 10:
                            continue

                        plate_roi = frame[py1:py2, px1:px2]
                        plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                        plate_gray = cv2.resize(plate_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                        # OCR with whitelist and psm
                        plate_text = pytesseract.image_to_string(
                            plate_gray,
                            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                        )
                        plate_text = ''.join(filter(str.isalnum, plate_text))

                        if plate_text and plate_text not in detected_plates:
                            detected_plates.add(plate_text)
                            cv2.putText(frame, plate_text, (px1, py2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # -------------------------------
        # Write and display frame
        # -------------------------------
        out.write(frame)
        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        progress.progress(min(1.0, frame_counter / total_frames))

    # -------------------------------
    # Release resources
    # -------------------------------
    cap.release()
    out.release()
    st.success(f"✅ Video processing completed! Saved as {out_path}")
    st.video(out_path)
    st.write("*Total Vehicle Counts:*", total_counts)
    st.write("*Detected License Plates:*", list(detected_plates))