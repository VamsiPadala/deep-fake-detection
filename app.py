import streamlit as st
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from joblib import load
from utils import detect_face
import tempfile

# Load model
model = load("model/model.pkl")

# Extract LBP Features
def extract_lbp_features(image):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# Prediction logic for image
def predict_image(image):
    face = detect_face(image)
    if face is None:
        return "No face detected"
    features = extract_lbp_features(face).reshape(1, -1)
    prediction = model.predict(features)[0]
    return "Fake" if prediction else "Real"

# Prediction logic for video
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Could not read video"
    face = detect_face(frame)
    if face is None:
        return "No face detected"
    features = extract_lbp_features(face).reshape(1, -1)
    prediction = model.predict(features)[0]
    return "Fake" if prediction else "Real"

# UI
st.set_page_config(page_title="Deepfake Detection System", layout="centered")

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0;">
        <h1 style="color:#4B0082; font-size:36px; margin-bottom: 5px;">Deepfake Detection System</h1>
        <h3 style="color:#555;">Acharya Nagarjuna University</h3>
        <h4 style="color:#888;">Mentor: DR.U Satish Kumar Sir</h4>
        <p style="color:#999;">Team: Vennela | Vamsi | Deepthi | Venkatesh</p>
    </div>
    <hr style="border: 1px solid #4B0082;">
    """,
    unsafe_allow_html=True
)

st.markdown("### Upload Image or Video for Deepfake Detection")
uploaded_file = st.file_uploader("Choose a media file (image or .mp4 video)", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(file_bytes)
        temp_path = temp.name

    if uploaded_file.type.startswith("image"):
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        result = predict_image(image)
        st.success(f"Prediction: {result}")

    elif uploaded_file.type == "video/mp4":
        st.video(file_bytes)
        result = predict_video(temp_path)
        st.success(f"Prediction: {result}")
