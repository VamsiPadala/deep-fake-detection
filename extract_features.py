import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from utils import detect_face

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def process_videos(folder, label):
    features = []
    labels = []

    for file in os.listdir(folder):
        if not file.endswith(".mp4"):
            continue
        path = os.path.join(folder, file)
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            face = detect_face(frame)
            if face is not None:
                feat = extract_lbp_features(face)
                features.append(feat)
                labels.append(label)
            else:
                print(f"[WARN] No face detected in: {path}")
        else:
            print(f"[ERROR] Couldn't read video: {path}")
    
    return features, labels

if __name__ == "__main__":
    os.makedirs("features", exist_ok=True)  # Create folder if it doesn't exist
    X, y = [], []

    real_features, real_labels = process_videos("dataset/real", 0)
    fake_features, fake_labels = process_videos("dataset/fake", 1)

    X.extend(real_features + fake_features)
    y.extend(real_labels + fake_labels)

    np.save("features/X.npy", X)
    np.save("features/y.npy", y)

    print(f"[INFO] Feature extraction completed. Total samples: {len(X)}")
