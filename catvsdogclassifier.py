# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 19:15:33 2019

@author: Prateek_Sharma
"""
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation,BatchNormalization

# Initialising the CNN
catvsdogclassifier = Sequential()

# Step 1 - Convolution
catvsdogclassifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
catvsdogclassifier.add(BatchNormalization())

# Step 2 - Pooling
catvsdogclassifier.add(MaxPooling2D(pool_size = (2, 2)))
catvsdogclassifier.add(Dropout(0.25))


# Adding a second convolutional layer
catvsdogclassifier.add(Conv2D(64, (3, 3), activation = 'relu'))
catvsdogclassifier.add(BatchNormalization())
catvsdogclassifier.add(MaxPooling2D(pool_size = (2, 2)))
catvsdogclassifier.add(Dropout(0.25))

# Adding a third convolutional layer
catvsdogclassifier.add(Conv2D(128,(3,3),activation='relu'))
catvsdogclassifier.add(BatchNormalization())
catvsdogclassifier.add(MaxPooling2D(pool_size=(2,2)))
catvsdogclassifier.add(Dropout(0.25))

# Step 3 - Flattening
catvsdogclassifier.add(Flatten())

# Step 4 - Full connection
catvsdogclassifier.add(Dense(128, activation = 'relu'))
catvsdogclassifier.add(BatchNormalization())
catvsdogclassifier.add(Dropout(0.5))
catvsdogclassifier.add(Dense(1, activation = 'sigmoid'))

# Compiling the CNN
catvsdogclassifier.compile(loss='binary_crossentropy',
  optimizer='adam',metrics=['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Loading the training Set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
# Loading the test Set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Training the classifier
catvsdogclassifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 27,
                         validation_data = test_set)

# Converting the Model to json
catvsdog_json = catvsdogclassifier.to_json()
with open("./catvsdog.json","w") as json_file:
  json_file.write(catvsdog_json)

# Saving the weights seperately
catvsdogclassifier.save_weights("./catvsdog.h5")
