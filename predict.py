import cv2
import numpy as np
import joblib
from utils import detect_face
from skimage.feature import local_binary_pattern

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

model = joblib.load("model/rf_model.pkl")

def predict(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Error reading video."

    face = detect_face(frame)
    if face is None:
        return "No face detected."

    feat = extract_lbp_features(face)
    result = model.predict([feat])
    return "Fake" if result[0] == 1 else "Real"
