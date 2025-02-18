import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define the directories for the training dataset
train_dir = 'C:/Users/user/Desktop/PPROJECT/implementation/train_reduced'
img_size = (128, 128)  # Ensure the image size used for training

# Function to load images and labels from the dataset
def load_data(train_dir):
    images = []
    labels = []
    for label in os.listdir(train_dir):
        label_dir = os.path.join(train_dir, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, image_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
                img = cv2.resize(img, img_size)  # Resize to fixed size
                images.append(img.flatten())  # Flatten and store
                labels.append(label)
    return np.array(images), np.array(labels)

# Load data
X, y = load_data(train_dir)

# Normalize the images
X = X / 255.0

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label encoder classes for later use in inference
np.save('label_classes.npy', label_encoder.classes_)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define your deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(128, 128, 1), input_shape=(128*128,)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Save the training history
with open('history.pkl', 'wb') as file:
    pickle.dump({
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }, file)

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.show()

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('model_loss.png')
plt.show()

# Save the trained model
model.save('C:/Users/user/Desktop/PPROJECT/deep_learning_model.h5')
