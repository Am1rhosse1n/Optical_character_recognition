# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from plot_history import plot_history
#==============================================================================    
#load Data Set
(trainImage, trainLable) , (testImage, testLable) = mnist.load_data()

#Image reshape : change image to vector for feed to Conv network
xTrain = trainImage.reshape(60000,28,28,1)
xTest = testImage.reshape(10000,28,28,1)

# normaling
xTrain = xTrain.astype('float32')
xTrain = xTrain/255
xTest = xTest.astype('float32')
xTest = xTest/255

#lable preparing
from keras.utils import np_utils
yTrain = np_utils.to_categorical(trainLable)
yTest = np_utils.to_categorical(testLable)

#Creating Our Model Model(Functional)
from keras.models import Model
from keras.layers import Conv2D , MaxPool2D , Input,  Flatten , Dense
import keras

myInput = Input(shape = (28,28,1))
conv1 = Conv2D(16,(3,3), activation='relu' , padding='same')(myInput)
pool1 = MaxPool2D(pool_size=2)(conv1)
#bejaye estefade az pooling mishavad anra hazf kard va az stride estefade kard k joze voroodihaye laye conv ast
conv2 = Conv2D(32,(3,3), activation='relu' , padding='same')(pool1)
pool2 = MaxPool2D(pool_size=2)(conv2)
flat = Flatten()(pool2)
out_layer = Dense(10 , activation='softmax' )(flat)

myModel = Model(myInput, out_layer)

myModel.summary()
myModel.compile(optimizer= keras.optimizers.Adam() ,loss = keras.losses.categorical_crossentropy ,metrics = ['accuracy'])

#Training
netHistory = myModel.fit(xTrain,yTrain, batch_size = 200 , epochs=20,validation_split=0.2)
plot_history(netHistory)

#Evaluation
testLoss , testAcc =myModel.evaluate(xTest,yTest)
labels_predict = myModel.predict(xTest)
labels_predict = np.argmax(labels_predict,axis=1)