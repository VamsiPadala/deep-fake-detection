# Deep Fake Recognition using lightweight ResNet18 (PyTorch)

**Project Title:** Deep Fake Recognition using lightweight ResNet18 model with PyTorch

**Team:**

* P. Vamsi (L23CS3270)
* A. Vennela Priya (Y22CS3202)
* K. Deepthi Vimal (L23CS3268)
* K. Venkatesh (Y22CS3226)

**Institution / Guide:**

* Acharya Nagarjuna University
* Under the guidance of Dr. U. Satish Kumar
* Principle: Prof. Lingaraju

---

## 🚀 Project Overview

This repository contains code and documentation for a deepfake detection system that classifies videos as **real** or **fake** using facial feature extraction and a lightweight **ResNet18** model implemented in **PyTorch**. The system is designed for academic usage, quick experimentation, and deployment as a web/UI application for real-time or batch inference.

### Key ideas

* Extract face frames from videos using OpenCV and landmark/face detectors.
* Compute feature representations (face crops, embeddings) and feed them to a ResNet18 backbone.
* Train a lightweight classifier head for binary real/fake classification.
* Provide a simple UI to test videos and visualize predictions.

---

## ✨ Features

* Face extraction pipeline (frame sampling, detection, alignment)
* Lightweight ResNet18 backbone (PyTorch) for efficient training and inference
* Training scripts with checkpoints and TensorBoard logging
* Evaluation scripts for accuracy, precision, recall, F1-score, ROC-AUC
* Inference script for single video or directory of videos
* Simple web UI (Flask / Streamlit) to upload and test videos (optional)

---

## 📁 Recommended repository structure

```
deepfake-resnet18/
├─ data/
│  ├─ raw/                 # raw videos (train/val/test)
│  ├─ frames/              # extracted frames organized by video
│  └─ metadata/            # csv/json with labels and splits
├─ notebooks/              # EDA and visualization notebooks
├─ src/
│  ├─ data_utils.py        # dataset and preprocessing utilities
│  ├─ face_extractor.py    # OpenCV face detection & alignment
│  ├─ model.py             # ResNet18 based model definition
│  ├─ train.py             # training loop
│  ├─ eval.py              # evaluation scripts
│  └─ infer.py             # inference script
├─ web_app/                # optional: Flask or Streamlit UI
├─ checkpoints/            # saved models and weights
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/deepfake-resnet18.git
cd deepfake-resnet18
```

2. Create a Python virtual environment and activate it (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**Suggested `requirements.txt` (examples)**

```
torch>=1.8.0
torchvision
opencv-python
numpy
pandas
scikit-learn
tqdm
tensorboard
flask       # if using Flask web app
streamlit   # if using Streamlit web app
facenet-pytorch # optional: for face embeddings
```

---

## 🧰 Dataset

> *You must provide or point to your dataset.*

* Place raw videos in `data/raw/` with a CSV `data/metadata/labels.csv` containing columns: `video_id, filepath, label` where label is `real` or `fake` (or 0/1).
* Recommended public datasets for benchmarking: FaceForensics++, Celeb-DF, DFDC (if license allows).
* Make sure you respect dataset licenses and privacy rules.

---

## 🔄 Preprocessing / Feature Extraction

1. Extract frames from videos (e.g., sample 1 frame per second or select N frames per video).
2. Detect faces per frame using OpenCV DNN, Haar cascades, MTCNN, or other detectors.
3. Align and crop face regions to a fixed size (e.g., 224×224) and save to `data/frames/`.

Example usage:

```bash
python src/face_extractor.py --input_dir data/raw --output_dir data/frames --fps 1
```

---

## 📐 Model

* Backbone: ResNet18 (pretrained on ImageNet) with final classifier modified for binary output.
* Implemented in `src/model.py`. Use `torchvision.models.resnet18(pretrained=True)` and replace `fc` layer.

Example snippet (high-level):

```python
from torchvision import models
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # binary
```

---

## 🏋️ Training

Train with `src/train.py`. Key features of the training script:

* Dataset loader with transforms and augmentation
* Option to resume from checkpoint
* Learning rate scheduler, early stopping
* TensorBoard logging

Example command:

```bash
python src/train.py --data-dir data/frames --labels data/metadata/labels.csv --epochs 30 --batch-size 32 --lr 1e-4 --checkpoint-dir checkpoints/
```

---

## 🔍 Evaluation

Evaluate on held-out test set using `src/eval.py`.
Outputs: classification report (precision/recall/F1), confusion matrix, ROC curve and AUC.

Example:

```bash
python src/eval.py --checkpoint checkpoints/best.pth --data-dir data/frames --labels data/metadata/test_labels.csv
```

---

## ▶️ Inference

To run inference on a single video:

```bash
python src/infer.py --checkpoint checkpoints/best.pth --video-path data/raw/example_video.mp4 --output results/output.mp4
```

The script should:

* Sample frames
* Extract faces
* Run model per-frame
* Aggregate frame-level predictions to a video-level score (mean, median or majority vote)
* Overlay predicted label on output video

---

## 🧪 UI / Demo (optional)

A minimal Flask/Streamlit app is included in `web_app/` which allows uploading a video and returns a prediction and visualization. Start with:

```bash
cd web_app
streamlit run app.py    # or: python app.py for Flask
```

---

## 📈 Results & Metrics

Add your final reported results here (accuracy, precision, recall, F1, AUC) and example screenshots of the UI or sample outputs.

---

## 📝 Notes & Best Practices

* Balance dataset classes or use class weights in the loss if dataset is imbalanced.
* Carefully separate training and test sources to avoid identity leakage.
* Use cross-validation or multiple test splits for robust evaluation.
* Consider temporal models (LSTM, 3D CNN) or attention on top of frame features for improved performance.

---

## 🤝 Contributing

Contributions are welcome. Please open an issue or submit a pull request with a clear description of changes and tests.

---

## 📜 License

This repository is released under the **MIT License** — see `LICENSE` for details.

---

## 📚 References

* FaceForensics++
* DFDC (Deepfake Detection Challenge)
* Relevant papers on DeepFake detection and ResNet architectures

---

## 📞 Contact

For questions, reach out to the team lead:

* **P. Vamsi** — [padalavamsi@zohomail.com](mailto:padalavamsi@zohomail.com)

**Good luck — and thank you for contributing to responsible research on deepfake detection!**
