import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import os

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Command line argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

# Function to plot accuracy and loss curves
def plot_model_history(model_history, filename='plot.png'):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # Summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1, max(1, len(model_history.history['accuracy']) // 10)))
    axs[0].legend(['train', 'val'], loc='best')
    # Summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1, max(1, len(model_history.history['loss']) // 10)))
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig(filename)
    plt.show()

# Define data directories
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 75 # Increased epochs slightly, EarlyStopping will manage it

# Data Augmentation for training images (same as in dataset_prepare.py for consistency)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,          # Rotate images by up to 10 degrees
    width_shift_range=0.1,      # Shift images horizontally by up to 10%
    height_shift_range=0.1,     # Shift images vertically by up to 10%
    shear_range=0.1,            # Shear transformations
    zoom_range=0.1,             # Zoom in/out by up to 10%
    horizontal_flip=True,       # Flip images horizontally
    fill_mode='nearest'         # Fill in new pixels created by transformations
)

# Only rescale for validation images
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale", # Use grayscale as input for custom CNN
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale", # Use grayscale as input for custom CNN
        class_mode='categorical')

# --- Optimized Custom CNN Model ---
model = Sequential()

# Input Block
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 4
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# Flatten and Dense Layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax')) # Output layer for 7 emotions

# --- Training Mode ---
if mode == "train":
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy']) # Adjusted learning rate

    # Callbacks for better training
    callbacks = [
        # Reduce learning rate when validation loss stops improving
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=1e-7, verbose=1),
        # Stop training if validation loss doesn't improve for a certain number of epochs
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
    ]
    
    print("Starting optimized Custom CNN training...")
    model_info = model.fit(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size,
            callbacks=callbacks)
    
    plot_model_history(model_info, filename='plot_custom_cnn_final.png')
    # Save the entire model (architecture + weights)
    model.save('emotion_model_custom_final.h5') 
    print("Optimized Custom CNN model trained and saved as 'emotion_model_custom_final.h5'")

# --- Display Mode ---
elif mode == "display":
    # Load the entire model
    model_path = 'emotion_model_custom_final.h5'
    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found. Please train the model first by running with --mode train.")
        exit()
        
    model = load_model(model_path)

    cv2.ocl.setUseOpenCL(False)

    # Dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure a webcam is connected and not in use by another application.")
        exit()

    # Load Haar Cascade Classifier
    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if facecasc.empty():
        print("Error: Could not load face cascade classifier. Make sure 'haarcascade_frontalface_default.xml' is in the correct path or downloaded.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, exiting...")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w] # Use grayscale for custom CNN input
            
            # Resize and preprocess for custom CNN
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0) # Add batch and channel dimension
            cropped_img = cropped_img / 255.0 # Rescale pixels to [0, 1]

            # Make prediction
            prediction = model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1280,720),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




