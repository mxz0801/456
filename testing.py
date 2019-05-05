
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import model_from_json
import os
import pandas as pd
import numpy as np

#Test the test dataset using pre-trained model and weights
test_data_dir='../chest/chest_xray/test/'
test_datagen = ImageDataGenerator(rescale = 1./255)   #rescaling images

#generate testing dataset images
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary')


#load trained model from json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("test.h5")
print("Loaded model from disk")

#learning rate and decay from previous training
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-5)


#load trained weight 
loaded_model.compile(loss='binary_crossentropy', 
                     optimizer=opt, 
                     metrics=['accuracy'])

#output result
score = loaded_model.evaluate_generator(test_generator)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))