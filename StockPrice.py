#---------------------------------------------------------------------------------------------------
import random                                                                                      #
import time                                                                                        #
import datetime as dt                                                                              #
import pandas as pd                                                                                #
from pandas import read_csv                                                                        #
import numpy as np                                                                                 #
import matplotlib.pyplot as plt                                                                    #
#---------------------------------------------------------------------------------------------------
import math                                                                                        #
import matplotlib.pyplot as plt                                                                    #
#---------------------------------------------------------------------------------------------------
from keras.models import Sequential                                                                #
from keras.layers import LSTM, Embedding, Dense, Dropout, RepeatVector, TimeDistributed            #
from keras.optimizers import Adam                                                                  #
#---------------------------------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler                                                     #
from sklearn.metrics import mean_squared_error                                                     #
#---------------------------------------------------------------------------------------------------
import requests                                                                                    #
import pandas_datareader.data as web                                                               #
import pandas_datareader as pdr                                                                    #
#-------------------------------------------- API tools --------------------------------------------
from flask import Flask, jsonify                                                                   #
#---------------------------------------------------------------------------------------------------

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
#--------------------------------------------------------------------------------------------------------------
# Get dataset via YAHOO API
df = pdr.get_data_yahoo('TSLA', start="2010-1-1")
dataset = np.array(df[["Close"]])
print(dataset[-20:])
dataset = dataset.astype('float32')
#--------------------------------------------------------------------------------------------------------------
# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
#--------------------------------------------------------------------------------------------------------------
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#--------------------------------------------------------------------------------------------------------------
# reshape into X=t and Y=t+1
look_back = 6
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
#--------------------------------------------------------------------------------------------------------------
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#--------------------------------------------------------------------------------------------------------------
model = Sequential()
model.add(LSTM(60, input_shape=(1, look_back),return_sequences=True))#(4, input_shape=(1, look_back)))
model.add(LSTM(60, activation='relu',return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(1))
model.compile(Adam(0.001),loss='mean_squared_error')
h =model.fit(trainX, trainY, epochs=10, batch_size=3, verbose=2, validation_split = 0.05)
#--------------------------------------------------------------------------------------------------------------
# make predictions
print("trainX shape = ", trainX.shape)
print("testX  shape = ", testX.shape)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
#--------------------------------------------------------------------------------------------------------------
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
#--------------------------------------------------------------------------------------------------------------
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
#--------------------------------------------------------------------------------------------------------------
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#--------------------------------------------------------------------------------------------------------------
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
#--------------------------------------------------------------------------------------------------------------
print("Prediction")
print(np.array(df[["Close"]][-20:]))
print("Real Data")
print(testPredictPlot[-20:])
#--------------------------------------------------------------------------------------------------------------
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), "g-")
#--------------------------------------------------------------------------------------------------------------
plt.plot(trainPredictPlot, "b")
plt.plot(testPredictPlot, "r")
plt.legend(["dataset", "trainPredictPlot", "testPredictPlot"])
plt.show()
#--------------------------------------------------------------------------------------------------------------
#print("--------------------------------------------------------------------------------------------------------------")
#print("Model Summary")
#model.summary()
#--------------------------------------------------------------------------------------------------------------


randArr=[]
randArr= np.array(randArr)
randDF = np.array([])

while(True):
    #--------------------------------------------------------------------------------------------------------------
    # Get dataset via YAHOO API
    dfNew = pdr.get_data_yahoo('TSLA', start="2019-1-1")
    datasetNew = np.array(dfNew[["Close"]])
    #print("------------------------------------------------------------------------------")
    #print(datasetNew[-20:])
    datasetNew = datasetNew.astype('float32')
    closeCurrent = datasetNew
    #--------------------------------------------------------------------------------------------------------------
    # Normalize the datasetNew
    scaler = MinMaxScaler(feature_range=(0, 1))
    datasetNew = scaler.fit_transform(datasetNew)
    #--------------------------------------------------------------------------------------------------------------
    updatedData = datasetNew[-15:]
    look_backNew = 6
    prdctNew, _ = create_dataset(updatedData, look_backNew)
    #--------------------------------------------------------------------------------------------------------------
    #print(prdctNew.shape)
    #--------------------------------------------------------------------------------------------------------------
    prdctNew = np.reshape(prdctNew, (prdctNew.shape[0], 1, prdctNew.shape[1]))
    #--------------------------------------------------------------------------------------------------------------
    qwe = model.predict(prdctNew)
    PredictedValues = scaler.inverse_transform(qwe)
    #print(PredictedValues)          #("prediction = ",qwe)
    print("Last Prediction  =>  ",PredictedValues[-1:])          #("prediction = ",qwe)
    print("Last closeCurrent  =>  ",closeCurrent[-1:])          #("prediction = ",qwe)
    time.sleep(2)



