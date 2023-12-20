import easyocr
import streamlit as st
import torch
import cv2
from PIL import Image, ImageFilter
import io
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Initialize YOLO and EasyOCR
yolo = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'])

# Function to read license plate
def read_plate(img, yolo, reader, is_bgr=True):
    img = cv2.imread(img)
    results = yolo(img)
    results = results[0]
    box = results.boxes
    current_cordinates = [round(x) for x in box.xyxy.flatten().tolist()]
    (x1, y1, x2, y2) = current_cordinates
    plate_img = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 17, 15, 15)
    text = reader.readtext(blurred)
    text = ' '.join([t[1] for t in text])
    img = results.plot()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    text_position = (75, y2 - 20)  # Adjust the position as needed
    ax.text(*text_position, text, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the image with text
    st.pyplot(fig)
    st.write('The car plate number is:', text)

# Streamlit app
def main():
    st.title('Car Plate Detection and Recognition')
    st.write('This app uses YOLO and EasyOCR to detect and recognize car plates from images.')

    # Background styling
    st.markdown(
        """
        <style>
        .main {
            background-image: url("https://images.unsplash.com/photo-1603386329225-868f9b1ee6c9?q=80&w=2069&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }
        </style>,
        """, unsafe_allow_html=True)

    # Use st.session_state to store the state of the app
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'png', 'PNG', 'jpeg'])

    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        st.image(st.session_state.uploaded_image, caption='Uploaded Image.', use_column_width=True)

        # Add a selectbox for choosing the operation
        operation = st.selectbox('Choose Operation:', ['Read Plate', 'Blur', 'Sharpen'])

        if st.button('Process'):
            if operation == 'Read Plate':
                img_path = "temp.jpg"
                st.session_state.uploaded_image.save(img_path)
                read_plate(img_path, yolo, reader)
                
                # Reset uploaded image state after processing
                st.session_state.uploaded_image = None
                
            elif operation == 'Blur':
                # Add your blurring logic here
                blurred_img = st.session_state.uploaded_image.filter(ImageFilter.BLUR)
                st.image(blurred_img, caption='Blurred Image.', use_column_width=True)
            elif operation == 'Sharpen':
                # Add your sharpening logic here
                sharpened_img = st.session_state.uploaded_image.filter(ImageFilter.SHARPEN)
                st.image(sharpened_img, caption='Sharpened Image.', use_column_width=True)

if __name__ == "__main__":
    main()
