
# Facial Emotion Recognition

A deep learning-based real-time **Facial Emotion Recognition (FER)** system using a custom **Convolutional Neural Network (CNN)** trained on the **FER-2013 dataset**. Deployed using **Streamlit**, it enables live webcam-based detection of human emotions.

## 🧩 Problem It Solves
Detecting human emotions manually is:
- Time-consuming
- Subjective
- Inconsistent

This system **automates emotional state recognition** using computer vision and deep learning, enabling **real-time**, **non-intrusive**, and **scalable** emotion tracking, which is especially useful in human-computer interaction and behavioral analytics.

## 💡 Real-World Applications

- **E-learning**: Track student engagement during online classes.
- **Healthcare**: Early diagnosis of emotional disorders like depression.
- **Customer Service**: Improve chatbot & human response based on customer mood.
- **Driver Monitoring**: Detect fatigue or road rage.
- **Interactive Games**: Adapt game content based on player emotions.
- **Marketing**: Analyze emotional feedback from users during product testing.

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

## 📊 About the Dataset – FER-2013

**FER-2013 (Facial Expression Recognition 2013)** is a publicly available dataset introduced in the 2013 ICML Challenges. It consists of:
- **35,887 grayscale images** of size **48x48 pixels**
- **7 emotion classes**
- **CSV format**: each row includes an emotion label, pixel values, and usage (Train/PublicTest/PrivateTest)

### Why FER-2013?
- It is one of the largest labeled facial expression datasets.
- Contains **diverse**, **real-world**, and **low-resolution** faces collected via the Google.
- Excellent benchmark for facial emotion recognition research.

## 😐 Emotion Labels
The system classifies facial expressions into the following 7 categories:
- Angry 😠
- Disgusted 🤢
- Fearful 😨
- Happy 😀
- Neutral 😐
- Sad 😢
- Surprised 😲

---

## ✨ Preprocessing & Augmentation (`preprocessing.py`)

### 📌 What It Does:
- **Converts pixel strings** from the CSV into actual 48x48 grayscale images.
- **Organizes images** into a directory structure: `data/train/<emotion>/` and `data/test/<emotion>/`.
- **Balances the dataset** using **class-wise augmentation** only for underrepresented classes.

### 🧠 Why Augmentation?
Real-world emotion datasets are often **imbalanced**. For instance, "Happy" or "Neutral" emotions are overrepresented, while "Disgusted" and "Fearful" are scarce.

To overcome this:
- The **ImageDataGenerator** is configured with real-world transformations like rotation, zoom, shear, and horizontal flip.
- Underrepresented classes are augmented until all classes have equal representation—this helps prevent model bias.

```python
ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
```

This ensures better generalization and a robust model trained on **diverse and balanced data**.

---

## 🧠 CNN Architecture (`training.py`)

A **custom deep CNN** designed for 48x48 grayscale images:
- **4 Convolutional Blocks**: Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout
- Final layers: Flatten → Dense(512) → BatchNorm → Dropout → Output (Softmax)
- **Regularization**: Dropout layers and Batch Normalization
- **Callbacks**: EarlyStopping and ReduceLROnPlateau

Training is conducted with augmented data and validation split for generalization.

---

## 🌐 Streamlit UI (`streamlit_app.py`)

- Real-time webcam-based facial emotion recognition
- Uses **OpenCV Haarcascade** to detect faces
- Predicts emotion using the trained CNN and displays:
  - Live webcam feed
  - Predicted emotion label and confidence score
  - Bar chart of emotion probabilities

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Download & Place Dataset
Download FER-2013 from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place `fer2013.csv` into the `src/` folder.

### 3️⃣ Preprocess & Augment
```bash
cd src
python preprocessing.py
```

### 4️⃣ Train the Model
```bash
python training.py --mode train
```

### 5️⃣ Launch the Real-Time UI
```bash
streamlit run streamlit_app.py
```
