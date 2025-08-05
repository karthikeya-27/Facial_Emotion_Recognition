import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator # For augmentation

# convert string to integer
def atoi(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord("0")
    return n

# making folders
outer_names = ['test','train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
base_data_path = 'data' # Define base path for clarity

os.makedirs(base_data_path, exist_ok=True)
for outer_name in outer_names:
    for inner_name in inner_names:
        # Create 'data/test/angry', 'data/train/angry', etc.
        os.makedirs(os.path.join(base_data_path, outer_name, inner_name), exist_ok=True)

df = pd.read_csv('./fer2013.csv')
mat = np.zeros((48,48),dtype=np.uint8)
print("Saving original images and collecting train counts...")

# Dictionaries to store counts and image data for balancing
train_emotion_counts = Counter()
train_images_by_emotion = {emotion_idx: [] for emotion_idx in range(7)}

# Read the csv file line by line and save original images
# Also collect train image data for later augmentation
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    emotion_label = df['emotion'][i]
    
    # Convert pixel string to 48x48 numpy array
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat) # Create PIL Image from numpy array

    # train dataset (first 28709 images)
    if i < 28709:
        # Save original image to its respective train folder
        emotion_name = inner_names[emotion_label]
        current_count = train_emotion_counts[emotion_label]
        img_path = os.path.join(base_data_path, 'train', emotion_name, f'im_orig_{current_count}.png')
        img.save(img_path)
        
        # Store image data (as numpy array) for later augmentation
        train_images_by_emotion[emotion_label].append(mat.copy()) # Use .copy() to avoid issues with mat changing
        train_emotion_counts[emotion_label] += 1
    # test dataset (remaining images)
    else:
        # Save original image to its respective test folder
        emotion_name = inner_names[emotion_label]
        # Use separate counters for test set to avoid conflicts with train counts
        if 'test_counts' not in locals():
            test_counts = {name: 0 for name in inner_names}
        img_path = os.path.join(base_data_path, 'test', emotion_name, f'im_orig_{test_counts[emotion_name]}.png')
        img.save(img_path)
        test_counts[emotion_name] += 1

print("\nOriginal Training Emotion Distribution:")
for emotion_idx, count in train_emotion_counts.items():
    print(f"{inner_names[emotion_idx]}: {count}")

# --- Data Balancing with Augmentation ---
print("\nBalancing training dataset with augmentation...")

# Determine the target count for balancing (e.g., max count of any emotion in train set)
max_train_count = max(train_emotion_counts.values())
print(f"Target count for each emotion: {max_train_count}")

# Define a robust ImageDataGenerator for augmentation
# Note: We're augmenting grayscale images, so color_mode is not relevant here,
# but the transformations are.
augment_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

for emotion_idx, current_count in train_emotion_counts.items():
    emotion_name = inner_names[emotion_idx]
    images_to_augment = train_images_by_emotion[emotion_idx]
    
    if current_count < max_train_count:
        needed_images = max_train_count - current_count
        print(f"Augmenting '{emotion_name}': Need {needed_images} more images.")

        # Convert list of numpy arrays to a single numpy array for flow
        # Reshape to (num_images, height, width, channels)
        # For grayscale, channels=1
        images_array = np.array(images_to_augment).reshape(-1, 48, 48, 1)

        # Create an iterator for augmented images
        i = 0
        for batch in augment_datagen.flow(images_array, batch_size=1, shuffle=True):
            if i >= needed_images:
                break
            
            # Convert augmented image (numpy array) back to PIL Image (remove channel dim for saving)
            augmented_img_array = batch[0].reshape(48, 48) # Remove channel dimension
            augmented_img = Image.fromarray(augmented_img_array.astype(np.uint8))

            # Save the augmented image
            save_path = os.path.join(base_data_path, 'train', emotion_name, f'im_aug_{current_count + i}.png')
            augmented_img.save(save_path)
            i += 1
            # Update tqdm progress bar for this specific emotion
            if i % 100 == 0: # Print progress every 100 images
                print(f"  Saved {i}/{needed_images} augmented images for '{emotion_name}'")

print("\nDataset balancing complete!")
print("Final Training Emotion Distribution (after augmentation):")
final_train_counts = Counter()
for emotion_name in inner_names:
    final_train_counts[emotion_name] = len(os.listdir(os.path.join(base_data_path, 'train', emotion_name)))
for emotion_name, count in final_train_counts.items():
    print(f"{emotion_name}: {count}")

print("\nDone!")
