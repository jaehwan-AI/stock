# library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from slice_ojh import split_stock, split_mm
from sklearn.model_selection import train_test_split, TimeSeriesSplit

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV

# data load
x = np.load('x_data.npy', allow_pickle = 'True')
y = np.load('y_data.npy', allow_pickle = 'True')

# data preprocess
def prepro(data, method):    
    if method == 'stand':
        scaler = StandardScaler()
        x = data.reshape(-1,1)
        scaler.fit(x)
        tmp = scaler.transform(x)
        return np.squeeze(tmp, axis=1)

x_stand = np.zeros((x.shape[0], x.shape[1]))
for i in range(len(x[0])):
    x_stand[:,i] = prepro(x[:,i], 'stand')

# data slice
features, labels = split_stock(x_stand, 25, 1, 3)
print(features.shape, labels.shape)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, shuffle=False)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# LSTM model
def LSTM_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape= (25,5), name = 'input')
    x = LSTM(128, activation = 'relu', name = 'hidden1', return_sequences=True)(inputs)
    x = Dropout(drop)(x)
    x = LSTM(128, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(100, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(10, activation = 'relu', name = 'hidden4')(x)
    outputs = Dense(1, name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, loss = 'mse', metrics = ['mse'])
    return model

# parameter
def create_hyperparameters(): # epochs, node, acivation 추가 가능
    batches = [16, 32, 64]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]                           
    return {'batch_size' : batches, 'optimizer': optimizers, 
           'drop': dropout}

# wrapper
model = KerasRegressor(build_fn = LSTM_model, verbose = 2)

hyperparameters = create_hyperparameters()
time_cv = TimeSeriesSplit(n_splits=5).split(x_train)
search = RandomizedSearchCV(model, hyperparameters, cv = time_cv)

best_LSTM = search.fit(x_train, y_train, epochs=30, verbose=2)
best_param = search.best_params_
print(best_LSTM)
print(best_param)

best_pred = best_LSTM.predict(x_test)

# plotting
fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(best_pred, label='Prediction')
ax.legend()
plt.show()

# data preprocess
scaler = StandardScaler()
tmp = x[:,3].reshape(-1,1)
scaler.fit(tmp)
tmp = scaler.transform(tmp)

# today close
x_today = x_stand[-25:,:]
x_today = x_today[np.newaxis,:]
pred_today = best_LSTM.predict(x_today)
print(pred_today)
pred_today = pred_today[np.newaxis]
pred_today = scaler.inverse_transform(pred_today)
print(pred_today)










