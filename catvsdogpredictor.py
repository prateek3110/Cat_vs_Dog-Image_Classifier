# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 09:37:21 2019

@author: Prateek_Sharma
"""

# Importing the packages
from keras.models import model_from_json
import cv2
import numpy as np

# Loading the Model from Json File
json_file = open('./catvsdog.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the weights
loaded_model.load_weights("./catvsdog.h5")

# Compiling the model
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# loading the image we want to test
image = cv2.imread('./data/input_image.jpg')
image = cv2.resize(image, (128,128))
image = image.reshape(1, 128, 128, 3)

# Predict to which class your input image has been classified
result = loaded_model.predict_classes(image)
if(result[0][0] == 1):
    print("This is a Dog!")
else:
    print("This is a Cat!")