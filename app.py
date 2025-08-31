import streamlit as st
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")

st.title("YOLOv8 Object Detection Web App")
st.markdown(
    """
    Upload an image or select a sample image, then click **Run Detection** to see YOLOv8 predictions.
    The model detects objects and displays bounding boxes with confidence scores.
    """
)

# Load YOLOv8 model with safe globals context (fixes PyTorch 2.6+ unpickling error)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model


model = load_model()
if model is None:
    st.stop()

# Sidebar About section (optional)
with st.sidebar.expander("About this project"):
    st.markdown(
        """
        - **Dataset:** Custom dataset used for training
        - **Model:** YOLOv8s FP16 (best.pt)
        - **Accuracy:** ~95% mAP50
        - **Inference speed:** ~6 ms per image (on GPU)
        """
    )

# Image uploader and sample selector
uploaded_file = st.file_uploader("Upload an image (.jpg, .png)", type=["jpg", "png"])

sample_dir = "sample"
sample_images = []
if os.path.exists(sample_dir):
    sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png"))]

selected_sample = st.selectbox("Or select a sample image", [""] + sample_images)

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif selected_sample:
    image = Image.open(os.path.join(sample_dir, selected_sample)).convert("RGB")

if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

if st.button("Run Detection"):
    if image is None:
        st.warning("Please upload or select an image first.")
    else:
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        try:
            results = model(img_cv)
        except Exception as e:
            st.error(f"Error during inference: {e}")
            st.stop()

        # Draw bounding boxes and labels
        img_out = img_cv.copy()
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                label = model.names[cls] if model.names else str(cls)

                # Draw rectangle
                cv2.rectangle(img_out, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                # Draw label background
                text = f"{label} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_out, (xyxy[0], xyxy[1] - 20), (xyxy[0] + w, xyxy[1]), (0, 255, 0), -1)
                # Put label text
                cv2.putText(img_out, text, (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        img_out_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        st.image(img_out_rgb, caption="Detection Result", use_container_width=True)

        # Prepare detections table
        detections = []
        for r in results:
            for box in r.boxes:
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
