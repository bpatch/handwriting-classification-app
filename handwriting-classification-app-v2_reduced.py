import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from joblib import dump, load
from scipy.ndimage.interpolation import zoom
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.naive_bayes import MultinomialNB
import shutil

zoom_param = 0.25

zoom_int = int(100*zoom_param)

forest = load('forest')
cnn = load_model('cnn')
nb = load('nb')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


st.write("Draw a number between 0 and 9 in the black box and press guess.")

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
    new_image[new_image<0] = 0
    
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
        
else:


    col1, mid, col2 = st.columns([1,1,20])
    with col1:
        st.image('wizard_1.png', width=60)
    with col2:
        st.write('Wizard of the random forest guess:')
        
    col1, mid, col2 = st.columns([1,1,20])
    with col1:
        st.image('wizard_2.png', width=60)
    with col2:
        st.write('Naive Bayes wizard guess:')
        
    col1, mid, col2 = st.columns([1,1,20])
    with col1:
        st.image('wizard_3.png', width=100)
    with col2:
        st.write('Convoluted neurotic network wizard guess:')

# st.write("Make the wizards wiser by telling them what number you drew:")

# choice = st.radio("Number being drawn", [0,1,2,3,4,5,6,7,8,9])

# teach = st.button('Teach the wizards')
# if teach:
#     y_train = load('labels-saved')
#     X_train = load('features-saved')
#     new_image = zoom(rgb2gray(canvas_result.image_data), zoom_param).reshape(1, zoom_int ** 2)
#     new_image[new_image < 0] = 0
#     y_train.append(choice)
#     X_train = np.vstack([X_train, new_image])
#     dump(y_train, 'labels-saved')
#     dump(X_train, 'features-saved')
#     # Build RF classifier:
#     forest = RandomForestClassifier(random_state=1)

#     # Fit RF classifier:
#     forest.fit(X_train, y_train)

#     dump(forest , 'forest')

#     ############################
#     ### NAIVE BAYES APPROACH ###
#     ############################

#     # Build NB classifier:
#     nb = MultinomialNB()

#     # Fit RF classifier:
#     nb.fit(X_train, y_train)

#     dump(nb , 'nb')


#     ####################
#     ### CNN APPROACH ###
#     ####################

#     # Hyperparameter that could be tuned:
#     BATCH_SIZE = 5

#     # Reshape feature data for CNN approach:
#     X_train_cnn = X_train.reshape(X_train.shape[0], zoom_int, zoom_int)


#     # Split the data into training and test sets:
#     X_train_cnn, X_test_cnn, y_train, y_test = train_test_split(X_train_cnn,
#                                                                 y_train,
#                                                                 test_size=0.2,
#                                                                 shuffle=True,
#                                                                 stratify=y_train,
#                                                                 random_state=2021)

#     v_pixels = X_train_cnn.shape[1]
#     h_pixels = X_train_cnn.shape[2]

#     X_train_cnn2 = X_train_cnn.reshape((X_train_cnn.shape[0],
#                                         v_pixels,
#                                         h_pixels, 1)).astype('float32')

#     X_test_cnn2 = X_test_cnn.reshape((X_test_cnn.shape[0],
#                                       v_pixels,
#                                       h_pixels, 1)).astype('float32')

#     y_train_cnn = np_utils.to_categorical(y_train)
#     y_test_cnn = np_utils.to_categorical(y_test)
#     num_classes = y_test_cnn.shape[1]

#     # Create model
#     cnn = Sequential()
#     # Define input layer
#     cnn.add(Conv2D(32, (3, 3), input_shape=(v_pixels,
#                                             h_pixels, 1), activation='relu'))
#     # Define hidden layers
#     cnn.add(MaxPooling2D())
#     cnn.add(Dropout(0.2))
#     cnn.add(Flatten())
#     cnn.add(Dense(128, activation='relu'))
#     # Define output layer
#     cnn.add(Dense(num_classes, activation='softmax'))
#     # Compile model
#     cnn.compile(loss='categorical_crossentropy',
#                 optimizer='adam', metrics=['accuracy'])


#     # Fit the model
#     cnn.fit(X_train_cnn2, y_train_cnn, validation_data=(X_test_cnn2, y_test_cnn),
#               epochs=10, batch_size=BATCH_SIZE, verbose=0)
    
#     cnn.save('./cnn')
#     st.write("Training complete!")

    
    
    
    
    