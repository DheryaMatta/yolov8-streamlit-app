import streamlit as st
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os

# Streamlit page setup
st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("YOLOv8 Object Detection Web App")
st.markdown(
    """
    Upload an image or select a sample image, then click **Run Detection** to see YOLOv8 predictions.
    The model detects objects and displays bounding boxes with confidence scores.
    """
)

# Sidebar About section
with st.sidebar.expander("About this project"):
    st.markdown(
        """
        - **Dataset:** Custom dataset used for training
        - **Model:** YOLOv8s FP16 (best.pt)
        - **Accuracy:** ~95% mAP50
        - **Inference speed:** ~6 ms per image (on GPU)
        """
    )

# Load YOLO model safely (PyTorch >=2.6)
@st.cache_resource
def load_model():
    with torch.serialization.safe_globals([DetectionModel]):
        model = YOLO("best.pt")
    return model

model = load_model()

# Image uploader & sample selector
uploaded_file = st.file_uploader("Upload an image (.jpg, .png)", type=["jpg", "png"])

sample_images_dir = "samples"
sample_images = []
if os.path.exists(sample_images_dir):
    sample_images = [f for f in os.listdir(sample_images_dir) if f.lower().endswith((".jpg", ".png"))]

selected_sample = None
if sample_images:
    selected_sample = st.selectbox("Or select a sample image", [""] + sample_images)

# Helper functions to load images
def load_image_from_file(file) -> Image.Image:
    return Image.open(file).convert("RGB")

def load_image_from_path(path) -> Image.Image:
    return Image.open(path).convert("RGB")

image = None
if uploaded_file is not None:
    image = load_image_from_file(uploaded_file)
elif selected_sample:
    image = load_image_from_path(os.path.join(sample_images_dir, selected_sample))

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

run_detection = st.button("Run Detection")

# Draw bounding boxes and labels
def draw_boxes(image: np.ndarray, results) -> np.ndarray:
    for r in results:
        boxes = r.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = model.names[cls] if model.names else str(cls)
            cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (xyxy[0], xyxy[1] - 20), (xyxy[0] + w, xyxy[1]), (0, 255, 0), -1)
            cv2.putText(image, text, (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return image

# Run detection when button is clicked
if run_detection:
    if image is None:
        st.warning("Please upload or select an image first.")
    else:
        with st.spinner("Running YOLOv8 inference..."):
            img_np = np.array(image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            results = model(img_cv)
            img_out = draw_boxes(img_cv.copy(), results)
            img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
            st.image(img_out_rgb, caption="Detection Result", use_column_width=True)

            # Prepare table of detected objects
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = model.names[cls] if model.names else str(cls)
                    detections.append({"Class": label, "Confidence": conf})

            if detections:
                df = pd.DataFrame(detections).sort_values(by="Confidence", ascending=False).reset_index(drop=True)
                st.markdown("### Detected Objects")
                st.dataframe(df)
            else:
                st.info("No objects detected.")

