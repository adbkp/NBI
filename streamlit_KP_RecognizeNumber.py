import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import joblib
import requests
from io import BytesIO
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Corrected raw URL for the model file
model_url = "https://raw.githubusercontent.com/adbkp/NBI/NBI-AI/log_reg.pkl"

# Download the model
@st.cache_resource
def load_model():
    try:
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an error for bad responses
        return joblib.load(BytesIO(response.content), mmap_mode='r')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    


# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = image_array.reshape(1, -1)  # Flatten to 2D array
    return image_array

# Creating the Streamlit application
def main():
    st.sidebar.title("Navigation Menu")
    nav = st.sidebar.radio("Choose a Section", ["Purpose", "Number Recognition"])

    if nav == "Purpose":
        st.title("Number Recognition")
        st.header("Purpose")
        st.write("An application that recognizes handwritten numbers from images.")

    if nav == "Number Recognition":
        st.title("Number Recognition")
        st.write('Upload a PNG or JPEG file with a handwritten number, and the application will recognize it.')

        # Load the model
        model = load_model()
        if model is None:
            return

        # File uploader
        uploaded_file = st.file_uploader("Upload an image file", type=["jpeg", "jpg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            # Create two columns for image display
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.header("Processed Image")
                # Preprocess and display the resized grayscale image
                processed_image = image.convert("L").resize((28, 28))
                st.image(processed_image, use_container_width=True)

            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Make prediction with more detailed output
            try:
                # Get predictions and probabilities
                predictions = model.predict(preprocessed_image)
                
                # Try to get prediction probabilities if available
                try:
                    proba = model.predict_proba(preprocessed_image)[0]
                    # Show full probability distribution
                    st.write("Prediction Probabilities:")
                    prob_df = pd.DataFrame({
                        'Digit': range(10),
                        'Probability': proba
                    })
                    st.dataframe(prob_df)
                except AttributeError:
                    st.write("Probability distribution not available")

                # Show prediction details
                predicted_digit = predictions[0]
                st.write(f"**Raw Prediction:** {predicted_digit}")
                
                # Additional debugging information
                st.write("Preprocessed Image Array:")
                st.write(preprocessed_image)

            except Exception as e:
                st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()