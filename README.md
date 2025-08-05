# Facial Emotion Recognition System 🎭🤖

A deep learning project for real-time **Facial Emotion Recognition** using a custom-built **Convolutional Neural Network (CNN)** trained on the **FER-2013 dataset**, and deployed using **Streamlit** for live webcam-based detection.

---
## 📁 Project Structure
```bash
FACIAL_EMOTION_RECOGNITION/
│
├── Images/                            # (Optional) for saved architecture/plots
├── src/
│   ├── data/                          # Auto-generated: Contains organized & augmented training/test images
│   ├── emotion_model.h5               # Trained CNN model (output of training.py)
│   ├── fer2013.csv                    # Dataset in CSV format
│   ├── haarcascade_frontalface_default.xml # Haarcascade for face detection
│   ├── preprocessing.py               # Dataset preprocessor + augmenter
│   ├── training.py                    # CNN model trainer + visualizer
│   └── streamlit_app.py               # Real-time prediction app using webcam
│
├── requirements.txt                   # Python dependencies
```

---
## Emotion Labels
This project classifies 7 emotions:
- Angry
- Disgusted
- Fearful
- Happy
- Neutral
- Sad
- Surprised

---
## 🔍 Unique Features & Improvements

### Custom Preprocessing & Augmentation (`preprocessing.py`)
- Reads **FER-2013 CSV** format and converts pixel data into grayscale `.png` images.
- Organizes data into a clean folder structure under `data/train` and `data/test`.
- Performs **class-wise augmentation** using `ImageDataGenerator` to handle **imbalanced datasets**.
- Uses **dynamic augmentation**: augmenting only underrepresented classes to match the max class size.

### Deep CNN Architecture
- **4 Convolutional Blocks** with:
  - Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout
- Final Dense layers:
  - Flatten → Dense(512) → BatchNorm → Dropout → Output(7 classes)
- **Regularization techniques**: Dropout, Batch Normalization

### Real-Time UI with Streamlit (`streamlit_app.py`)
- Uses OpenCV + Haarcascade to detect faces from webcam.
- Runs prediction and overlays emotion + probability bar chart live.
- Fully interactive & easy to use with Streamlit UI elements.
- Auto error handling if camera or model fails.

---
## ▶️ How to Run the Project

### 1️⃣ Clone the Repository & Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Download Dataset
Get the FER-2013 dataset and place `fer2013.csv` in `src/`.

### 3️⃣ Preprocess and Augment Dataset
```bash
cd src
python preprocessing.py
```
This will:
- Create folders under `data/train` and `data/test`.
- Balance classes using augmentation.

### 4️⃣ Train the CNN Model
```bash
python training.py --mode train
```
- Trains model for up to 75 epochs (with early stopping).
- Saves model as `emotion_model.h5`.

### 5️⃣ Launch Streamlit App (Webcam-based UI)
```bash
streamlit run streamlit_app.py
```
This opens a browser-based app with:
- Live webcam feed
- Predicted emotion
- Bar chart of confidence scores

