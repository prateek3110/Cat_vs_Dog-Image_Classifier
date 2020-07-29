# Cat_vs_Dog-Image_Classifier
An Image Classifier that classifies an image as either cat or dog.
1. Training data and test data are obtained from https://www.kaggle.com/c/dogs-vs-cats/data to local storage.
2. CNN Sequential Model is created and compiled using the steps in the file 'catvsdogclassifier.py'.
3. The training and test data is reshaped to the shape which our model expects while training.
4. Model is trained on training data.
5. The trained weights and trained model are stored and saved in a file named 'catvsdog.h5' and 'catvsdog.json' respectively.
6. Once the model is trained and trained weights are saved, we use the trained model and trained weight from 'catvsdog.h5' and 'catvsdog.json' respectively and use them to predict whether the given image is cat or dog in the file named 'catvsdogpredictor.py.

Note : Spyder is used to develop the codes.While using it set the directory correctly.
