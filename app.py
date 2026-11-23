import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("lung_cancer_model.h5")

classes = ["Benign","Malignant","Normal"]

st.title("Lung Cancer Detection System")

file = st.file_uploader("Upload Lung Scan", type=["jpg","png"])

if file:

    image = Image.open(file)
    st.image(image)

    img = np.array(image)
    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    pred = model.predict(img)

    st.write("Prediction:", classes[np.argmax(pred)])