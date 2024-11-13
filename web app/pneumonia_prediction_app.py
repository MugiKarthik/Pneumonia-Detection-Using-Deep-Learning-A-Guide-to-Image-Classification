import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('C:/Users/KARTHIK M/Documents/College works/M.Sc Data Science/deep learning/project/model_weights/Xception_model.keras')

# Define the target size for the image (224x224 for Xception model)
target_size = (224, 224)

# Function to process and predict the image
def predict_image(img):
    # Resize the image to the target size
    img_resized = cv2.resize(img, target_size)

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalize the image (same preprocessing as used during training)
    img_array = np.expand_dims(img_rgb, axis=0) / 255.0

    # Predict the class probabilities
    prediction = model.predict(img_array)

    # The model output is a 2D array: [[probability_normal, probability_pneumonia]]
    # Find the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction)

    # Define class names
    class_names = ['NORMAL', 'PNEUMONIA']

    # Get the predicted class using the index
    predicted_class = class_names[predicted_class_index]

    return predicted_class, prediction, img_rgb

# Streamlit UI
st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to predict if it is normal or shows signs of pneumonia.")

# Upload image via Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image using PIL
    img = Image.open(uploaded_file)

    # Convert image to numpy array
    img = np.array(img)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Make predictions
    predicted_class, prediction, img_rgb = predict_image(img)

    # Display prediction
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Prediction probabilities: Normal: {prediction[0][0]:.4f}, Pneumonia: {prediction[0][1]:.4f}")

    # Plot the image with the predicted class
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.axis('off')  # Hide axes
    st.pyplot(fig)  # Pass the figure to st.pyplot
