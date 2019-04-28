
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
import os
import numpy as np

os.environ['PYTHONHASHSEED'] = "0"
np.random.seed(1)

train_data_dir='../chest/chest_xray/train/'
validation_data_dir='../chest/chest_xray/val/'
test_data_dir='../chest/chest_xray/test/'

num_train_size = 300
num_validation_size = 67
epochs = 20
batch_size = 12

img_width, img_height = 64, 64

input_shape=(img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same",
                 input_shape=(64,64,3),kernel_initializer = keras.initializers.glorot_uniform(seed=None)))
model.add(Conv2D(32,(3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3), activation="relu", padding="same",kernel_initializer = keras.initializers.glorot_uniform(seed=None)))
model.add(Conv2D(filters=32,kernel_size=(3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3), activation="relu", padding="same",kernel_initializer = keras.initializers.glorot_uniform(seed=None)))
model.add(Conv2D(filters=64,kernel_size=(3,3), activation="relu", padding="same"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-5)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)  

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=num_validation_size)

model.save_weights('test.h5')
scores=model.evaluate_generator(validation_generator)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

