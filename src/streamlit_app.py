import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import pandas as pd # Import pandas for bar chart data

# Suppress TensorFlow logging messages (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Streamlit UI Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Real-time Emotion Detector", layout="wide")

# --- Configuration ---
# Ensure this matches the model saved by your emotions.py script
MODEL_PATH = 'emotion_model_custom_final_v4.h5' 
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_DICT = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# --- Load Model and Cascade Classifier ---
@st.cache_resource # Cache the model loading for efficiency
def load_emotion_model():
    """Loads the pre-trained Keras emotion model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        st.stop()
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_resource # Cache the cascade loading
def load_face_cascade():
    """Loads the OpenCV Haar cascade classifier for face detection."""
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        st.error(f"Error: Haar cascade file '{CASCADE_PATH}' not found or could not be loaded.")
        st.stop()
    return face_cascade

model = load_emotion_model()
face_cascade = load_face_cascade()

# --- Streamlit UI Elements ---
st.title("Live Facial Emotion Recognition")

# Create an empty placeholder for instructions that can be cleared later
instruction_placeholder = st.empty()
if not st.session_state.get('webcam_running', False):
    with instruction_placeholder.container():
        st.markdown("""
            ### Get Started
            1. Click 'Start Webcam' to begin real-time emotion detection.
            2. Ensure good lighting and face the camera clearly.
            3. The detected emotion and its probability will be displayed.
            4. A live bar chart will show the confidence for each emotion.
        """)
        st.markdown("---") # Separator

# Create two columns for layout: one for video, one for probabilities
col_video, col_probs = st.columns([2, 1]) # 2:1 ratio for video and prob charts

with col_video:
    st.subheader("Webcam Feed")
    video_placeholder = st.empty() # Placeholder for the video frame
    status_text = st.empty() # Placeholder for status messages (e.g., "Webcam running")

with col_probs:
    st.subheader("Emotion Probabilities")
    prob_chart_placeholder = st.empty() # Placeholder for the bar chart
    dominant_emotion_placeholder = st.empty() # Placeholder for dominant emotion text

# Start/Stop button logic
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False

st.markdown("<br>", unsafe_allow_html=True) # Add some space
col1_btn, col2_btn = st.columns(2)
with col1_btn:
    start_button = st.button("Start Webcam", help="Click to start your webcam and begin detection.")
with col2_btn:
    stop_button = st.button("Stop Webcam", help="Click to stop the webcam feed.")

if start_button:
    st.session_state.webcam_running = True
    instruction_placeholder.empty() # Clear instructions when webcam starts
    st.rerun() # Rerun to update state and start video

if stop_button:
    st.session_state.webcam_running = False
    st.rerun() # Rerun to update state and stop video

if st.session_state.webcam_running:
    status_text.info("Webcam is running. Adjust your face to the center of the frame.")
    
    cap = cv2.VideoCapture(0) # Open default webcam
    if not cap.isOpened():
        status_text.error("Could not open webcam. Please ensure it's connected and not in use by another application.")
        st.session_state.webcam_running = False
        st.stop()

    while st.session_state.webcam_running:
        ret, frame = cap.read()
        if not ret:
            status_text.warning("Failed to grab frame. Exiting webcam feed.")
            st.session_state.webcam_running = False
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Initialize prediction data for the chart (if no face detected, show zeros)
        emotion_probs = {EMOTION_DICT[i]: 0.0 for i in range(len(EMOTION_DICT))}
        display_emotion_on_video = "No Face Detected"
        display_text_color = (0, 0, 255) # Red for no face

        if len(faces) > 0:
            # Take the first detected face for prediction
            x, y, w, h = faces[0] 
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2) # Blue rectangle

            # Extract face ROI and preprocess for model prediction
            roi_gray = gray_frame[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0) # Add batch and channel dim
            cropped_img = cropped_img / 255.0 # Rescale pixels

            # Make prediction
            prediction = model.predict(cropped_img, verbose=0)[0] # Get the first (and only) prediction array
            max_prob = np.max(prediction)
            maxindex = int(np.argmax(prediction))
            predicted_emotion = EMOTION_DICT[maxindex]

            # Populate emotion probabilities for the chart
            for i, prob in enumerate(prediction):
                emotion_probs[EMOTION_DICT[i]] = prob

            # Always show the top class, remove "Uncertain" logic
            display_emotion_on_video = f"{predicted_emotion} ({max_prob:.2f})"
            display_text_color = (255, 255, 255) # White for confident prediction

            # Put text on frame
            cv2.putText(frame, display_emotion_on_video, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, display_text_color, 2, cv2.LINE_AA)
        else:
            # No face detected, update text on video
            cv2.putText(frame, display_emotion_on_video, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, display_text_color, 2, cv2.LINE_AA)


        # Convert OpenCV BGR image to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame in Streamlit
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update the probability bar chart
        df_probs = pd.DataFrame(emotion_probs.items(), columns=['Emotion', 'Probability'])
        
        # Sort for consistent display in the bar chart
        df_probs = df_probs.sort_values(by='Probability', ascending=False)
        
        prob_chart_placeholder.bar_chart(df_probs.set_index('Emotion'))
        
        # Update dominant emotion text
        if len(faces) > 0:
            dominant_emotion_placeholder.write(f"**Dominant Emotion:** {display_emotion_on_video}")
        else:
            dominant_emotion_placeholder.write("**Dominant Emotion:** No Face Detected")

    cap.release()
    status_text.success("Webcam stopped.")
