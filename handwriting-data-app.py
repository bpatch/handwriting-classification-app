# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 00:07:10 2021

@author: brend
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from joblib import dump
from scipy.ndimage.interpolation import zoom

zoom_param = 0.25

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


st.title('Handwriting Data Generator')

# Streamlit runs from top to bottom on every iteraction so
# we check if `count` has already been initialized in st.session_state.

choice = st.radio("Number being drawn", [0,1,2,3,4,5,6,7,8,9])

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


if 'labels' not in st.session_state:
	st.session_state.labels = []

if 'features' not in st.session_state:
	st.session_state.features = np.zeros((1, int(100*zoom_param) ** 2))

if 'counts' not in st.session_state:
	st.session_state.counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

# Create a button which will increment the counter
add = st.button('Add data')
if add:
    new_image = zoom(rgb2gray(canvas_result.image_data), zoom_param).reshape(1, int(100*zoom_param) ** 2)
    st.session_state.labels.append(choice)
    st.session_state.features = np.vstack([st.session_state.features, new_image])
    st.session_state.counts[choice] += 1           
    
st.write(st.session_state.counts)

# st.write(st.session_state.labels)

# st.write(st.session_state.features)

save = st.button('Save data')
if save:
    dump(st.session_state.features[1:, :], 'features-saved')
    dump(np.array(st.session_state.labels), 'labels-saved')
    
