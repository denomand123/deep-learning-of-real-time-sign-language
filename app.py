import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

# Load the trained model
model = tf.keras.models.load_model('C:/Users/user/Desktop/PPROJECT/deep_learning_model.h5')

# Load label classes
label_classes = np.load('label_classes.npy')

def preprocess_image(image_data):
    # Convert base64 to image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = image.convert('L')  # Convert image to grayscale
    image = image.resize((128, 128))  # Resize image to match model input
    image = np.array(image)
    image = image.reshape(1, 128, 128, 1)  # Reshape for model input
    return image.astype(np.float32) / 255.0

def main():
    st.title('Real-Time Sign Language Recognition')
    
    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        # Convert image to base64 for processing
        img_bytes = uploaded_file.read()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Prepare data for model
        image_data = img_base64
        image = preprocess_image(image_data)
        
        # Predict using the model
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction[0])
        sign = label_classes[predicted_class]

        # Show result
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write(f'Prediction: {sign}')

if __name__ == "__main__":
    main()
