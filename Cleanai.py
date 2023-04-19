import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import requests
import streamlit as st
import subprocess
from PIL import Image
from io import BytesIO
import PIL.Image
import urllib
import tempfile


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Input


# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model



st.set_page_config(page_title="Clean AI", page_icon=":sunrise:", layout="wide")
# ---- MAINPAGE ----
st.title(":sunrise: Denoise Images")
st.write("On this page, artificial intelligence removes noise from photos, The file format supports JPEG, PNG and JPG")
st.markdown("_____")
st.write("## Remove Noise from your image")
st.write(
    ":sunrise: Try uploading an noise image to watch the noise magically removed. Full quality images can be downloaded from the sidebar. :grin:"
)
st.sidebar.write("## Upload and download :gear:")


# define the function to convert the image to bytes
def convert_image(img):
    buf = BytesIO()
    plt.imsave(buf, img, cmap='gray')
    buf.seek(0)
    return buf.read()

# prepare function
def process_image(image_path):
    img = cv2.imread(image_path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (540, 420))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.0
    img = np.reshape(img, (420, 540, 1))
    return img

model1 = load_model('Models/Model2.h5') # load first model
model2 = load_model('Models/Model1.h5') # load Second model


def fix_image(upload):
    image_path = upload
    col1.write("Noisy Image :camera:")
    img_noise = process_image(image_path)
    col1.image(img_noise)
    fixed = model2.predict(np.expand_dims(img_noise, axis=0), batch_size=36)
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed.squeeze()), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(my_upload.read())
        temp_file_path = temp_file.name
    fix_image(upload=temp_file_path)
else:
    image_path = "0001-USPS-dmm300_608.pdf-15.png"
    col1.write("Noisy Image :camera:")
    img_noise = process_image(image_path)
    col1.image(img_noise)
    fixed = model2.predict(np.expand_dims(img_noise, axis=0), batch_size=36)

    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed.squeeze()), "fixed.png", "image/png")


hide_st_style = """
            <style>
            #ManiMenu {visibility: hidden;}
            footer  {visibility: hidden;}
            header  {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html= True)
