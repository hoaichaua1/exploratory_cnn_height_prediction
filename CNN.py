import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
DATA_DIR = "db_synthetic_1"  # Update with your actual data directory
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Resize images to this size

# Read children heights from the text file (assuming it's a CSV with columns: scene, height)
children_heights_df = pd.read_csv(os.path.join(DATA_DIR, "children_heights.csv"))

# Initialize lists to store image paths and corresponding heights
image_paths = []
heights = []

# Loop through image files
for scene_number in range(200):
    for rotation_angle in [30, 60, 90, 120, 150]:
        image_filename = f"scene{scene_number:03d}_{rotation_angle:03d}.png"
        image_path = os.path.join(DATA_DIR, image_filename)
        height = children_heights_df.loc[children_heights_df["scene"] == scene_number, "height"].values[0]
        image_paths.append(image_path)
        heights.append(height)

# Convert lists to numpy arrays
image_paths = np.array(image_paths)
heights = np.array(heights)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_paths, heights, test_size=0.2, random_state=42)

# Load and preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return img_array

X_train_processed = np.array([preprocess_image(image_path) for image_path in X_train])
X_test_processed = np.array([preprocess_image(image_path) for image_path in X_test])

# Create a simple CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(X_train_processed, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the testing set
test_loss = model.evaluate(X_test_processed, y_test)
print(f"Test loss (mean squared error): {test_loss:.4f}")

# Make predictions
sample_image_path = X_test[0]
sample_image_array = preprocess_image(sample_image_path)
predicted_height = model.predict(np.array([sample_image_array]))[0][0]
print(f"Predicted height for {sample_image_path}: {predicted_height:.2f} cm")




