import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="ISL Recognition", page_icon=":handshake:")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('ISL_model.h5')

try:
    model = load_my_model()
except:
    st.error("Error loading the model.")
    st.stop()

LABELS = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 
    7: "I", 8: "K", 9: "L", 10: "M", 11: "N", 12: "O", 13: "P", 
    14: "Q", 15: "R", 16: "S", 17: "T", 18: "U", 19: "V", 20: "W", 
    21: "X", 22: "Z"
}

st.title("Indian Sign Language Recognition")
st.write("Upload an image of a hand gesture.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=400)
    
    img_array = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_array, (128, 128))
    img_normalized = img_resized / 255.0
    img_final = np.expand_dims(img_normalized, axis=0)

    predictions = model.predict(img_final)
    score = np.max(predictions)
    class_idx = np.argmax(predictions)
    label = LABELS.get(class_idx, "Unknown")

    if score > 0.45:
        st.success(f"### Predicted Sign: **{label}**")
    else:
        st.warning("Low confidence prediction. Please try another image.")
