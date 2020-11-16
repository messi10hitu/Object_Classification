#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from glob import glob
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class objectclassify:
    def __init__(self, filename):
        self.filename = filename

    def predictionobjectclassify(self):
        # load model
        model = keras.models.load_model('TF_BAG_WATCH_GLASS_SHOE_Resnet50.h5')

        # summarize model
        # model.summary()
        imagename = self.filename

        # load an image from file
        image = load_img(imagename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model.predict(image)
        # print(yhat)
        # print(np.argmax(yhat[0]))

        classification = [np.argmax(yhat[0])]
        # print(classification)
        if classification == [0]:
            prediction = 'backpack'
            return [{"image": prediction}]
        elif classification == [1]:
            prediction = 'footwear'
            return [{"image": prediction}]
        elif classification == [2]:
            prediction = 'glasses'
            return [{"image": prediction}]
        elif classification == [3]:
            prediction = 'watch'
            return [{"image": prediction}]
