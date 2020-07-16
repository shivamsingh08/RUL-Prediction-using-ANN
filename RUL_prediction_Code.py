#!/usr/bin/env python

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data=pd.read_csv("RUL_data.csv")

list(data.columns)

training=data.iloc[1:200,]
testing=data.iloc[200:333,]

# fit scaler on training data
normalised = MinMaxScaler().fit(training.iloc[:,1:28])

# transform training data
X_train_scaled = normalised.transform(training.iloc[:,1:28])

# transform testing dataabs
X_test_scaled = normalised.transform(testing.iloc[:,1:28])


Y_train=training.iloc[:,28]

Y_test=testing.iloc[:,28]

params = {
    "epochs": [10, 20],
    "batch_size": [10,20,40],
    "n_layers": [1, 2],
    "n_neurons": [20, 40, 60],
    "dropout": [0.1, 0.2],
    "optimizers": ["nadam", "adam"],
    "activations": ["relu", "sigmoid"],
    "last_layer_activations": ["relu"],
    "losses": ["mean_squared_error"],
    "metrics": ["MSE"]
}


#Using the NeuroEvaluation package to predict the best parameters for applyting GA to the model.
from evolution import NeuroEvolution


search = NeuroEvolution(generations = 10, population = 10, params=params)

search.evolve(X_train_scaled, Y_train, X_test_scaled, Y_test)

#Building the ANN model

import keras
from keras.models import Sequential
from keras.layers import Dense


mod = Sequential()
mod.add(Dense(60, input_dim=27, activation='relu'))
mod.add(Dense(60, activation='relu'))
mod.add(Dense(60,  activation='relu'))



mod.add(Dense(1, activation='relu'))

mod.compile(loss='mean_squared_error', optimizer='nadam', metrics=['MSE'])

history = mod.fit(X_train_scaled, Y_train, epochs=20, batch_size=40)

predicted=mod.predict(X_test_scaled)

predicted=pd.Series(predicted.reshape(-1))


#Saving the file

pred.to_csv("RUL_pred_values.csv")


#Evaluating the Model

from sklearn.metrics import mean_squared_error
from math import sqrt


RMSE_h11 = sqrt(mean_squared_error(round(pred[18:23]),Y_test[18:23]))
RMSE_h12 = sqrt(mean_squared_error(round(pred[39:44]),Y_test[39:44]))
RMSE_h13 = sqrt(mean_squared_error(round(pred[84:89]),Y_test[84:89]))
RMSE_h14 = sqrt(mean_squared_error(round(pred[128:133]),Y_test[128:133]))


import statistics
print("Median RMSE:")
statistics.median([RMSE_h11,RMSE_h12,RMSE_h13,RMSE_h14])






