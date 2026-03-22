import streamlit as st
import cv2
import numpy as np
import os
import json
import torch
from PIL import Image
from ultralytics import YOLO

# Project imports
import ocr_engine

# Ensure directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

# Page config
st.set_page_config(
    page_title="Instinct GPT OCR - Meter Reading",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #1e1e1e;
        border: 1px solid #333;
        margin-bottom: 20px;
    }
    .metric-val {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff4b4b;
    }
    .metric-label {
        font-size: 1rem;
        color: #aaa;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚡ Settings")
    conf_threshold = st.slider("Detection Confidence", 0.01, 1.0, 0.15, 0.01)
    st.info("Using YOLOv8s 200-Epoch Optimized Model")
    
    st.divider()
    st.markdown("### Model Stats")
    st.write("- **Architecture:** YOLOv8s")
    st.write("- **Dataset size:** 619 images")
    st.write("- **Epochs:** 200")
    st.write("- **mAP50:** 0.963")

    st.divider()
    with st.expander("🛠️ Debug Diagnostics"):
        st.write(f"**CWD:** `{os.getcwd()}`")
        model_path = "runs/detect/meter_detector4/weights/best.pt"
        exists = os.path.exists(model_path)
        st.write(f"**Model Path:** `{model_path}`")
        st.write(f"**Found:** {'✅' if exists else '❌'}")
        if exists:
            # Check size
            st.write(f"**Size:** {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

# Main UI
st.title("📟 Smart Meter OCR Dashboard")
st.subheader("Extract precise readings from utility meters using advanced computer vision.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Meter Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save temp file
        img_path = os.path.join("tmp", "upload.jpg")
        os.makedirs("tmp", exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width="stretch")
        
        if st.button("🚀 Run Inference"):
            with st.spinner("Processing meter reading..."):
                # Define output path
                result_json_path = os.path.join("outputs", "st_result.json")
                
                # Execute inference
                try:
                    ocr_engine.execute_inference(img_path, result_json_path)
                    
                    if os.path.exists(result_json_path):
                        with open(result_json_path, 'r') as f:
                            result = json.load(f)
                        st.session_state['latest_result'] = result
                        st.success("Analysis Complete!")
                    else:
                        st.error("Inference did not generate a result file.")
                except Exception as e:
                    st.error(f"Inference Error: {e}")

with col2:
    if 'latest_result' in st.session_state:
        res = st.session_state['latest_result']
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-label">Detected Reading</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-val">{res["raw_text"]}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-label">Confidence Score</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-val">{res["confidence"]:.2f}</div>', unsafe_allow_html=True)
        
        if res["reject_to_qc"]:
            st.warning("⚠️ Low confidence detected. Quality Control review recommended.")
        else:
            st.success("✅ High confidence result.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display debug images
        st.divider()
        st.write("#### Pipeline Visualizations")
        tabs = st.tabs(["Target Field", "Dewarped", "Enhanced"])
        
        with tabs[0]:
            if os.path.exists("outputs/debug_target.jpg"):
                st.image("outputs/debug_target.jpg", width="stretch")
        with tabs[1]:
            if os.path.exists("outputs/debug_warped.jpg"):
                st.image("outputs/debug_warped.jpg", width="stretch")
        with tabs[2]:
            if os.path.exists("outputs/debug_enhanced.jpg"):
                st.image("outputs/debug_enhanced.jpg", width="stretch")
    else:
        st.info("Upload an image and click 'Run Inference' to see results.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Built by team GPT</p>
</div>
""", unsafe_allow_html=True)
