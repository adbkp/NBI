import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import streamlit as st
from PIL import Image
import joblib
import requests
from io import BytesIO

from scipy.ndimage import center_of_mass
from sklearn.datasets import fetch_openml


# Corrected raw URL for the model file
model_url = "https://raw.githubusercontent.com/adbkp/NBI/NBI-AI/my_model.pkl"



# Download the model
@st.cache_resource
def load_model():
    try:
        # Alternativ 1: Använd lokal fil
        # return joblib.load('mnist_model.pkl')
        
        # Alternativ 2: Fortsätt med URL men använd en ny tränad modell
        # model_url = "URL_TO_NEW_MODEL"
        response = requests.get(model_url)
        response.raise_for_status()
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
    
    # Add debugging information
    st.write("Image array shape before reshape:", image_array.shape)
    st.write("Min value:", image_array.min(), "Max value:", image_array.max())
    
    # Normalize to [0, 1] as per MNIST standard
    image_array = image_array / 255.0
    
    # Reshape to (1, 784) for model input
    image_array = image_array.reshape(1, -1)
    
    # Add more debugging
    st.write("Final array shape:", image_array.shape)
    st.write("Final min value:", image_array.min(), "Final max value:", image_array.max())
    
    return image_array

# Funktion för att spara MNIST-exempel
def save_mnist_samples():
    # Skapa en mapp för exempel om den inte finns
    sample_dir = "mnist_samples"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # Hämta några exempel från MNIST
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Spara de första 10 bilderna (en av varje siffra)
    for i in range(10):
        # Hitta första förekomsten av varje siffra
        idx = np.where(y == str(i))[0][0]
        
        # Reshape bilden till 28x28
        img = X[idx].reshape(28, 28)
        
        # Spara bilden
        plt.imsave(
            os.path.join(sample_dir, f'mnist_sample_{i}.png'),
            img,
            cmap='gray'
        )
    return sample_dir

# Creating the Streamlit application . 
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
        
        # Add model information
        st.write("Model information:")
        st.write("Model type:", type(model).__name__)
        try:
            st.write("Number of features expected:", model.n_features_in_)
            st.write("Classes:", model.classes_)
        except AttributeError:
            st.write("Could not get model details")

        # Add option to use sample MNIST image
        use_sample = st.checkbox("Use sample MNIST image")
        
        if use_sample:
            # Skapa/använd MNIST-exempel
            sample_dir = save_mnist_samples()
            
            # Låt användaren välja vilken siffra att testa
            sample_digit = st.selectbox(
                "Select a sample digit to test",
                range(10)
            )
            
            # Ladda vald exempel-bild
            sample_path = os.path.join(sample_dir, f'mnist_sample_{sample_digit}.png')
            image = Image.open(sample_path)
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

        if model is not None:
            # Test prediction with known MNIST sample
            try:
                X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
                test_sample = X[0:1] / 255.0  # Ta första MNIST-exemplet
                test_prediction = model.predict(test_sample)
                st.write(f"Test prediction on known MNIST sample (should be {y[0]}): {test_prediction[0]}")
            except Exception as e:
                st.error(f"Test prediction failed: {e}")

if __name__ == "__main__":
    main()