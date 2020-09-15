#!/usr/bin/env python
# coding: utf-8

# In[1]:


# library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from slice_ojh import split_stock, split_mm
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV

import re
import warnings
warnings.filterwarnings('ignore')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA


# In[2]:


x = np.load('x_data.npy', allow_pickle = 'True')


# In[3]:


# data preprocess
def prepro(data, method):
    if method == 'minmax':
        scaler = MinMaxScaler()
        x = data.reshape(-1,1)
        scaler.fit(x)
        tmp = scaler.transform(x)
        return np.squeeze(tmp, axis=1)
    
    if method == 'stand':
        scaler = StandardScaler()
        x = data.reshape(-1,1)
        scaler.fit(x)
        tmp = scaler.transform(x)
        return np.squeeze(tmp, axis=1)
    
    if method == 'robust':
        scaler = RobustScaler()
        x = data.reshape(-1,1)
        scaler.fit(x)
        tmp = scaler.transform(x)
        return np.squeeze(tmp, axis=1)


# ### Xgboost

# In[4]:


# data preprocess
scaler = StandardScaler()
tmp = x[:,3].reshape(-1,1)
scaler.fit(tmp)
tmp = scaler.transform(tmp)
x_stand = np.squeeze(tmp, axis=1)

# close data
features, labels = split_mm(x_stand, 25, 1)
print(features.shape, labels.shape)

# split data
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, shuffle=False)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# training Xgboost
xgb = XGBRegressor(learning_rate=0.01, n_estimators=100, max_depth=5, random_state=3)
xgb_fit = xgb.fit(x_train, y_train)

# predict
xgb_pred = xgb.predict(x_test)

# plotting
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(xgb_pred, label='Prediction')
ax.legend()
plt.show()


# In[5]:


# Xgboost hyperparameter
params = {
    'learning_rate':[0.01,0.05,0.1,0.15,0.2],
    'n_estimators':[50,100,150,200],
    'max_depth':[3,5,7,9],
    'subsample':[0.6,0.8,1.0],
    'min_child_weight':[1,5,10],
    'gamma':[0.5,1,1.5,2],
    'max_delta_step':[1,3,5,7,9],
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
    'colsample_bytree':[0.6,0.7,0.8,0.9,1.0]}
time_cv = TimeSeriesSplit(n_splits=5).split(x_train)
search = RandomizedSearchCV(xgb, params, cv=time_cv)
best_xgb = search.fit(x_train, y_train, verbose=2)
best_param = search.best_params_
print(best_xgb)
print(best_param)

best_pred = best_xgb.predict(x_test)

# plotting
fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(best_pred, label='Prediction')
ax.legend()
plt.show()


# In[6]:


# data preprocess
scaler = StandardScaler()
tmp = x[:,3].reshape(-1,1)
scaler.fit(tmp)
tmp = scaler.transform(tmp)
x_stand = np.squeeze(tmp, axis=1)


# In[7]:


# today close
x_today = x_stand[-25:]
x_today = x_today[np.newaxis,:]
tmp = best_xgb.predict(x_today)
print(tmp)
pred_today = scaler.inverse_transform(tmp)
print(pred_today)


# In[8]:


# atfer 2day
x_today2 = x_stand[-24:]
x_today2 = np.append(x_today2, tmp)
x_today2 = x_today2[np.newaxis,:]
tmp2= best_xgb.predict(x_today2)
print(tmp2)
pred_today2 = scaler.inverse_transform(tmp2)
print(pred_today2)


# In[9]:


# atfer 3day
x_today3 = x_stand[-23:]
x_today3 = np.append(x_today3, tmp)
x_today3 = np.append(x_today3, tmp2)
x_today3 = x_today3[np.newaxis,:]
tmp3 = best_xgb.predict(x_today3)
print(tmp3)
pred_today = scaler.inverse_transform(tmp3)
print(pred_today)


# In[10]:


# atfer 4day
x_today4 = x_stand[-22:]
x_today4 = np.append(x_today4, tmp)
x_today4 = np.append(x_today4, tmp2)
x_today4 = np.append(x_today4, tmp3)
x_today4 = x_today4[np.newaxis,:]
tmp3 = best_xgb.predict(x_today4)
print(tmp3)
pred_today = scaler.inverse_transform(tmp3)
print(pred_today)


# In[11]:


print(x_today.shape)
print(x_today)
print(x_today4)
print(x_today4.shape)


# ### LSTM

# In[12]:


# data preprocess
x_stand = np.zeros((x.shape[0], x.shape[1]))
for i in range(len(x[0])):
    x_stand[:,i] = prepro(x[:,i], 'stand')

# data slice
features, labels = split_stock(x_stand, 25, 3, 3)
print(features.shape, labels.shape)


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, shuffle=False)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[14]:


# LSTM model
def LSTM_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape= (25,5), name = 'input')
    x = LSTM(512, activation = 'relu', name = 'hidden1', return_sequences=True)(inputs)
    x = Dropout(drop)(x)
    x = LSTM(256, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(100, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(20, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    #x = Dense(20, activation = 'relu', name = 'hidden4')(x)
    #x = Dropout(drop)(x)
    outputs = Dense(3, name = 'output')(x)
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


# In[ ]:


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


# In[ ]:


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


# ### ARIMA

# In[ ]:


samsung = pd.read_csv('samsung0914.csv')
samsung = samsung[::-1]
samsung = samsung.drop(['시가','고가','저가','거래량'], axis=1)
samsung['종가'] = samsung['종가'].map(lambda x: int(re.sub(r',', '', x)))
samsung.reset_index(drop=True, inplace=True)
samsung.columns = ['date', 'close']

samsung['date'] = pd.to_datetime(samsung.date, format='%Y-%m-%d')
samsung = samsung.set_index('date')
print(samsung.info())


# In[ ]:


plot_acf(samsung) # ACF plot
plot_pacf(samsung) # PACF plot
plt.show()


# In[ ]:


y = samsung['close']
y_1diff = samsung.diff().dropna()['close']
result = adfuller(y)
print(f'원 데이터 ADF Statistic : {result[0] : .4f}')
print(f'원 데이터 p-value : {result[1] : .4f}')
result = adfuller(y_1diff)
print(f'1차 차분 ADF Statistic : {result[0] : .4f}')
print(f'1차 차분 p-value : {result[1] : .4f}')


# In[ ]:


model = ARIMA(samsung, order=(1,1,0)) # freq='D'
model_fit = model.fit()
print(model_fit.summary())


# In[ ]:


# rolling forecast
history = [x for x in samsung['close']]
predictions = []
for i in range(3):
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    history.append(yhat)

predictions
#plt.plot(test.index, test['close'])
#plt.plot(test.index, predictions, color='red')
#plt.show()


# In[ ]:


def auto_arima(data, order, sort='AIC'):
    order_lst = []
    aic_lst = []
    for p in range(order[0]):
        for d in range(order[1]):
            for q in range(order[2]):
                model = ARIMA(data, order=(p,d,q))
                try:
                    model_fit = model.fit()
                    c_order = f'p{p} d{d} q{q}'
                    aic = model_fit.aic
                    order_lst.append(c_order)
                    aic_lst.append(aic)
                except:
                    pass
    result_auto = pd.DataFrame(list(zip(order_lst, aic_lst)), columns=['order', 'AIC'])
    result_auto.sort_values(sort, inplace=True)
    return result_auto

result = auto_arima(samsung, [3,3,3])
result


# In[ ]:


# rolling forecast
history = [x for x in samsung['close']]
predictions = []
for i in range(3):
    model = ARIMA(history, order=(0,2,1))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    history.append(yhat)

predictions
#plt.plot(test.index, test['close'])
#plt.plot(test.index, predictions, color='red')
#plt.show()


# In[ ]:


#samsung.index[-1] + pd.Timedelta(days=1)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script 0914_stock_predict.ipynb')

