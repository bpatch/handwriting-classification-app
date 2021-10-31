import pandas as pd
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage.interpolation import zoom
from joblib import load
from keras.models import load_model

zoom_param = 0.25

zoom_int = int(100*zoom_param)

forest = load('forest')
cnn = load_model('cnn')
nb = load('nb')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


st.write("Draw a number in the black box and press guess.")

canvas_result = st_canvas(
        fill_color="rgb(0, 0, 0)",  # Fixed fill color with some opacity
        # stroke_width="1, 25, 3",
        stroke_width = 15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        width=100,
        height=100,
        drawing_mode="freedraw",
        key="canvas",
)

st.write("The three wizards will guess what number you drew.")


guess = st.button('Guess')
if guess:
    new_image = zoom(rgb2gray(canvas_result.image_data), zoom_param).reshape(1, zoom_int ** 2)

    # Random forest
    y_predict_rf = forest.predict(new_image)
    
    # Naive Bayes
    y_predict_nb = nb.predict(new_image)

    # Convolutional neural network

    new_image_reshape_1 = new_image.reshape(1, zoom_int, zoom_int)

    new_image_reshape_2 = new_image.reshape((1, zoom_int, zoom_int, 1)).astype('float32')

    y_predict_cnn = np.argmax(cnn.predict([new_image_reshape_2]), axis=1)




    col1, mid, col2 = st.columns([1,1,20])
    with col1:
        st.image('wizard_1.png', width=60)
    with col2:
        st.write('Wizard of the random forest guess:')
        st.write(y_predict_rf[0])
        
    col1, mid, col2 = st.columns([1,1,20])
    with col1:
        st.image('wizard_2.png', width=60)
    with col2:
        st.write('Naive Bayes wizard guess:')
        st.write(y_predict_nb[0])
        
    col1, mid, col2 = st.columns([1,1,20])
    with col1:
        st.image('wizard_3.png', width=100)
    with col2:
        st.write('Convoluted neurotic network wizard guess:')
        st.write(y_predict_cnn[0])
        