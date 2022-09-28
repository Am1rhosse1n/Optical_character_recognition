# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
#==========================================
def plot_history(netHistory):
    history = netHistory.history
    acc = history['acc']
    loss = history['loss']
    val_acc = history['val_acc']
    val_loss = history['val_loss']
    plt.xlabel('epoches')
    plt.ylabel('Error')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss','val_loss'])
    
    plt.figure()
    
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['acc','val_acc'])
#==============================================================================    
#load Data Set
(trainImage, trainLable) , (testImage, testLable) = mnist.load_data()

#Image reshape : change image to vector for feed to FC network
xTrain = trainImage.reshape(60000,784)
xTest = testImage.reshape(10000,784)

# normaling
xTrain = xTrain.astype('float32')
xTrain = xTrain/255
xTest = xTest.astype('float32')
xTest = xTest/255

#lable preparing
from keras.utils import np_utils
yTrain = np_utils.to_categorical(trainLable)
yTest = np_utils.to_categorical(testLable)

#Creating Our Model Model(Sequential)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
myModel = Sequential()
myModel.add(Dense(500, activation='relu', input_shape=(784,)))
myModel.add(Dense(100, activation='relu'))
myModel.add(Dense(10, activation='softmax'))
myModel.summary()
myModel.compile(optimizer= SGD(lr = 0.001) ,loss = 'categorical_crossentropy',metrics = ['accuracy'])

#Training
netHistory = myModel.fit(xTrain,yTrain, batch_size = 200 , epochs=20,validation_split=0.2)
plot_history(netHistory)

#Evaluation
testLoss , testAcc =myModel.evaluate(xTest,yTest)
labels_predict = myModel.predict(xTest)
labels_predict = np.argmax(labels_predict,axis=1)