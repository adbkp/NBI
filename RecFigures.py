import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib
import requests
from io import BytesIO
from scipy.ndimage import center_of_mass, sobel

# Corrected raw URL for the model file
model_url = "https://raw.githubusercontent.com/adbkp/NBI/NBI-AI/my_model.pkl"

# Download the model
@st.cache_resource
def load_model():
    try:
        response = requests.get(model_url)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")
    
    # Resize to 20x20 pixels, preserving aspect ratio
    image.thumbnail((20, 20), Image.LANCZOS)
    
    # Create a 28x28 white background
    background = Image.new('L', (28, 28), color=255)
    
    # Paste the resized image onto the center of the background
    offset = ((28 - image.size[0]) // 2, (28 - image.size[1]) // 2)
    background.paste(image, offset)
    
    # Convert to numpy array and invert colors
    image_array = 255 - np.array(background)
    
    # Apply Sobel edge detection
    edges = sobel(image_array)
    image_array = np.where(edges > np.percentile(edges, 75), 255, 0)
    
    # Apply thresholding
    threshold = np.percentile(image_array, 5)
    image_array[image_array < threshold] = 0
    image_array[image_array >= threshold] = 255
    
    # Center the image
    cy, cx = center_of_mass(image_array)
    if not np.isnan(cy) and not np.isnan(cx):
        shift_x, shift_y = np.round(14 - cx).astype(int), np.round(14 - cy).astype(int)
        image_array = np.roll(image_array, shift_x, axis=1)
        image_array = np.roll(image_array, shift_y, axis=0)
    
    # Make a copy for display
    display_array = image_array.copy()
    
    # Normalize for model input
    model_input = image_array / 255.0
    model_input = model_input.reshape(1, -1)
    
    # Convert display array back to a PIL Image
    display_image = Image.fromarray(display_array.astype('uint8'))
    
    # Resize the display image to match the original image size
    display_image = display_image.resize((200, 200), Image.LANCZOS)
    
    return model_input, display_image

def main():
    st.sidebar.title("Navigation Menu")
    nav = st.sidebar.radio("Choose a Section", ["Purpose", "Number Recognition", "About"])

    if nav == "Purpose":
        st.title("Number Recognition")
        st.header("Purpose")
        st.write("An application that recognizes handwritten numbers from images.")

    if nav == "About":
        st.title("About")
        st.header("This application was created by Kerstin Palö, January 30, 2025")
        st.write("The model is Random Forest")

    if nav == "Number Recognition":
        st.title("Number Recognition")
        st.write('Upload a PNG or JPEG file with a handwritten number, and the application will recognize it.')

        model = load_model()
        if model is None:
            return

        uploaded_file = st.file_uploader("Upload an image file", type=["jpeg", "jpg", "png"])

        if uploaded_file is not None:
            try:
                # Read and resize the original image
                original_image = Image.open(uploaded_file)
                original_image = original_image.resize((200, 200), Image.LANCZOS)
                
                # Process the image
                preprocessed_array, processed_image = preprocess_image(original_image)
                
                # Create two columns
                col1, col2 = st.columns(2)
                
                # Display original image in left column
                with col1:
                    st.header("Original Image")
                    st.image(original_image, use_container_width=True)
                
                # Display processed image in right column
                with col2:
                    st.header("Processed Image")
                    st.image(processed_image, use_container_width=True)

                # Get predictions
                predictions = model.predict(preprocessed_array)
                predicted_digit = predictions[0]
                
                # Get probabilities if available
                try:
                    proba = model.predict_proba(preprocessed_array)[0]
                    st.write("Prediction Probabilities:")
                    prob_df = pd.DataFrame({
                        'Digit': range(10),
                        'Probability': proba
                    })
                    st.dataframe(prob_df)
                except AttributeError:
                    st.write("Probability distribution not available")

                st.write(f"**Raw Prediction:** {predicted_digit}")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.error(f"Error details: {type(e).__name__}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()