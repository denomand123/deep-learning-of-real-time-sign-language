import os
import tensorflow as tf

# Define paths
base_path = 'C:/Users/user/Desktop/PPROJECT'
train_data_dir = os.path.join(base_path, 'implementation/train_reduced')
test_data_dir = os.path.join(base_path, 'implementation/test_reduced')

def preprocess_image(file_path, label):
    file_path = tf.strings.as_string(file_path)
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [150, 150])
    image = image / 255.0  # Normalize the image
    return image, label

def load_dataset(data_dir, batch_size=32):
    class_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    class_names.sort()
    class_indices = dict(zip(class_names, range(len(class_names))))
    
    file_paths = []
    labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, file_name))
            labels.append(class_indices[class_name])
    
    file_paths = tf.constant(file_paths)
    labels = tf.constant(labels)
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.shuffle(buffer_size=1000)  # Adjust buffer size if needed
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

train_dataset = load_dataset(train_data_dir)
test_dataset = load_dataset(test_data_dir)

# Model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(os.listdir(train_data_dir)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_dataset,
    steps_per_epoch=406,  # Adjusted for new dataset size
    epochs=10,
    validation_data=test_dataset,
    validation_steps=31)  # Adjusted for test dataset size

# Save the model
model.save('model.keras')
