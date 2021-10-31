
"""

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.naive_bayes import MultinomialNB
import time
from joblib import dump, load


zoom_param = 0.25

zoom_int = int(100*zoom_param)

# First load the data:
y_train = load('labels-saved')
X_train = load('features-saved')
X_train[X_train < 0] = 0

##############################
### RANDOM FOREST APPROACH ###
##############################

# Build RF classifier:
forest = RandomForestClassifier(random_state=1)

# Fit RF classifier:
forest.fit(X_train, y_train)

dump(forest , 'forest')

############################
### NAIVE BAYES APPROACH ###
############################

tic = time.time()

# Build NB classifier:
nb = MultinomialNB()

# Fit RF classifier:
nb.fit(X_train, y_train)

dump(nb , 'nb')


####################
### CNN APPROACH ###
####################

# Hyperparameter that could be tuned:
BATCH_SIZE = 5

# Reshape feature data for CNN approach:
X_train_cnn = X_train.reshape(X_train.shape[0], zoom_int, zoom_int)


# Split the data into training and test sets:
X_train_cnn, X_test_cnn, y_train, y_test = train_test_split(X_train_cnn,
                                                            y_train,
                                                            test_size=0.2,
                                                            shuffle=True,
                                                            stratify=y_train,
                                                            random_state=2021)

v_pixels = X_train_cnn.shape[1]
h_pixels = X_train_cnn.shape[2]

X_train_cnn2 = X_train_cnn.reshape((X_train_cnn.shape[0],
                                    v_pixels,
                                    h_pixels, 1)).astype('float32')

X_test_cnn2 = X_test_cnn.reshape((X_test_cnn.shape[0],
                                  v_pixels,
                                  h_pixels, 1)).astype('float32')

y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]

# Create model
cnn = Sequential()
# Define input layer
cnn.add(Conv2D(32, (3, 3), input_shape=(v_pixels,
                                        h_pixels, 1), activation='relu'))
# Define hidden layers
cnn.add(MaxPooling2D())
cnn.add(Dropout(0.2))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
# Define output layer
cnn.add(Dense(num_classes, activation='softmax'))
# Compile model
cnn.compile(loss='categorical_crossentropy',
            optimizer='adam', metrics=['accuracy'])


# Fit the model
cnn.fit(X_train_cnn2, y_train_cnn, validation_data=(X_test_cnn2, y_test_cnn),
          epochs=10, batch_size=BATCH_SIZE, verbose=0)

cnn.save('cnn')
