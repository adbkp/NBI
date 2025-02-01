import numpy as np
import pandas as pd

import streamlit as st
from PIL import Image
import joblib
import requests
from io import BytesIO

from scipy.ndimage import center_of_mass


# Corrected raw URL for the model file
model_url = "https://raw.githubusercontent.com/adbkp/NBI/NBI-AI/my_model.pkl"



# Download the model
@st.cache_resource
def load_model():
    try:
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an error for bad responses
        return joblib.load(BytesIO(response.content))
    except Exception as e:
      st.error(f"Error loading model: {e}")
      return None
    


def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")
    
    # Resize to 28x28 pixels directly (MNIST standard size)
    image = image.resize((28, 28), Image.LANCZOS)
    
    # Convert to numpy array and invert colors (MNIST has white digits on black background)
    image_array = 255 - np.array(image)
    
    # Normalize to [0, 1] as per MNIST standard
    image_array = image_array / 255.0
    
    # Reshape to (1, 784) for model input
    image_array = image_array.reshape(1, -1)
    
    return image_array

# Creating the Streamlit application
def main():
    st.sidebar.title("Navigation Menu")
    nav = st.sidebar.radio("Choose a Section", ["Purpose", "Number Recognition","About"])

    if nav == "Purpose":
        st.title("Number Recognition")
        st.header("Purpose")
        st.write("An application that recognizes handwritten numbers from images.")

    if nav == "About":
        st.title("About")
        st.header("This application was created by Kerstin Palö, January 30, 2025")
        st.write("The model is Random Forest ")

    if nav == "Number Recognition":
        st.title("Number Recognition")
        st.write('Upload a PNG or JPEG file with a handwritten number, or use a sample MNIST image.')

        # Load the model
        model = load_model()
        if model is None:
            return

        # Add option to use sample MNIST image
        use_sample = st.checkbox("Use sample MNIST image")
        
        if use_sample:
            # Load and display a sample MNIST image
            # You'll need to add sample MNIST images to your project
            sample_image = Image.open("path_to_sample_mnist_image.png")
            uploaded_file = None
            image = sample_image
        else:
            # File uploader
            uploaded_file = st.file_uploader("Upload an image file", type=["jpeg", "jpg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
            else:
                image = None

        if image is not None:
            # Display the uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.header("Processed Image")
                # Display the processed image
                processed_image = image.convert("L").resize((28, 28))
                st.image(processed_image, use_container_width=True)

            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            try:
                # Get predictions and probabilities
                predictions = model.predict(preprocessed_image)
                
                try:
                    proba = model.predict_proba(preprocessed_image)[0]
                    # Show top 3 predictions with probabilities
                    st.write("Top 3 Predictions:")
                    top_3_idx = np.argsort(proba)[-3:][::-1]
                    for idx in top_3_idx:
                        st.write(f"Digit {idx}: {proba[idx]*100:.2f}%")
                    
                    # Show full probability distribution
                    st.write("\nFull Probability Distribution:")
                    prob_df = pd.DataFrame({
                        'Digit': range(10),
                        'Probability (%)': proba * 100
                    })
                    st.dataframe(prob_df.style.format({'Probability (%)': '{:.2f}'}))
                    
                except AttributeError:
                    st.write("Probability distribution not available")

                # Show final prediction
                predicted_digit = predictions[0]
                st.write(f"\n**Final Prediction:** {predicted_digit}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()