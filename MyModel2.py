import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Define constants
data_dir = 'synthetic'  # Specify the path to your data directory
image_size = (512, 512, 20)  # Image size with depth channel and multiple angles
num_classes = 1  # Number of output classes (child height)
batch_size = 32
epochs = 20

# Function to load images and labels
def load_data(data_dir):
    images = []
    labels = []

    for child_id in range(200):  # Child IDs from 000 to 199
        combined_image = None  # Initialize combined image for each child
        text_file_name = f"scene000{child_id:03d}_030.txt"
        for angle in [30, 60, 90, 120, 150]:
            # Create file name
            file_name = f"scene000{child_id:03d}_{angle:03d}.png"
            #print(file_name)

            # Load RGB image
            image_path = os.path.join(data_dir, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load RGB channels
            #image = cv2.resize(image, image_size[:2])  # Resize image

            # Load corresponding depth map
            depth_file = file_name.replace('scene', 'depth')
            depth_path = os.path.join(data_dir, depth_file)
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)  # Load depth map
            #depth = cv2.resize(depth, image_size[:2])  # Resize depth map

            # Stack depth as an additional channel
            combined_single_angle_image = np.dstack((image, depth))

            if combined_image is None:
                combined_image = combined_single_angle_image
            else:
                combined_image = np.concatenate((combined_image, combined_single_angle_image), axis=-1)

        images.append(combined_image)

        # Load corresponding label
        
        label_path = os.path.join(data_dir, text_file_name)
        with open(label_path, 'r') as f:
            label = float(f.read().strip().split(':')[1])
        labels.append(label)

    return np.array(images), np.array(labels)

# Load data
images, labels = load_data(data_dir)

# +
print(labels.shape)
first_image = images[0]
first_layer = first_image[:, :, 19:21]  # Extract the first layer (channel)

# Display the first layer
plt.imshow(first_layer)  # Assuming it's a grayscale image
plt.axis('off')  # Hide axes
plt.show()
# -

# Split data into training and testing sets
images_train, images_test, labels_train, labels_test = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# Define CNN model (updated input shape with batch normalization layers)
def create_model():
    inputs = layers.Input(shape=image_size)

    # Convolutional layers with batch normalization
    conv = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    #conv = layers.BatchNormalization()(conv)
    conv = layers.MaxPooling2D((2, 2))(conv)
    #conv = layers.Conv2D(64, (3, 3), activation='relu')(conv)
    #conv = layers.BatchNormalization()(conv)
    conv = layers.MaxPooling2D((2, 2))(conv)
    conv = layers.Conv2D(128, (3, 3), activation='relu')(conv)
    #conv = layers.BatchNormalization()(conv)
    conv = layers.MaxPooling2D((2, 2))(conv)
    flatten = layers.Flatten()(conv)

    # Fully connected layers
    dense1 = layers.Dense(256, activation='relu')(flatten)
    output = layers.Dense(num_classes)(dense1)

    model = models.Model(inputs=inputs, outputs=output)
    return model

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(images_train, labels_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# +
# Evaluate the model
loss = model.evaluate(images_test, labels_test)
print("Test Loss:", loss)
# Predict labels for test set
predictions = model.predict(images_test)

# Calculate accuracy
accuracy = np.mean(np.abs(predictions - labels_test))
print("Test Accuracy (MAE):", accuracy)

# +
import matplotlib.pyplot as plt

# Plot predictions vs labels_test
plt.figure(figsize=(8, 6))
plt.scatter(labels_test, predictions, color='blue')
plt.plot([min(labels_test), max(labels_test)], [min(labels_test), max(labels_test)], color='red', linestyle='--')
plt.title('Predictions vs Actual Labels')
plt.xlabel('Actual Labels')
plt.ylabel('Predictions')
plt.grid(True)
plt.show()
# -



import matplotlib.pyplot as plt
# Plot the distribution of MAE
plt.figure(figsize=(8, 6))
plt.hist(np.abs(predictions - labels_test), bins=20, edgecolor='k')
plt.title('Distribution of Absolute Error')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


