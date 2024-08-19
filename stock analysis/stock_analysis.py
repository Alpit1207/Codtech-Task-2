# import the necessary libraies

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from subprocess import check_output
import warnings
from pandas.plotting import lag_plot
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# read the dataset

df = pd.read_csv("C:/Users/alpit/OneDrive/Desktop/Codtech/stock analysis/HDFCBANK.csv")
print(df.head())

print(df.shape)
print(df.columns)

df = df.drop(["Symbol","Series", "Deliverable Volume","%Deliverble","Trades"],axis=1)
print(df.columns)

# view the graph for close price

df['Close'].plot()
plt.title("HDFC Closing Price")
plt.show()

# Comulative Return

dr = df.cumsum()
dr.plot()
plt.title('HDFC Cumulative Returns')
plt.show()

# view the plot for open 

plt.figure(figsize=(10,10))
lag_plot(df['Open'], lag=5)
plt.title('HDFC Autocorrelation plot')
plt.show()

print(df['Date'][5305])

# assignt the train and test data range 

train_data = df[0:int(len(df)*0.8)]
test_data = df[int(len(df)*0.8):]
plt.figure(figsize=(12,7))
plt.title('HDFC Prices')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(df['Open'], 'blue', label='Training Data')
plt.plot(test_data['Open'], 'green', label='Testing Data')
plt.xticks(np.arange(0,5305, 550), df['Date'][0:5305:550])
plt.legend()
plt.show()

# create a function to assess the accuracy of predictions

def smape_fun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))

# develop the ARIMA model for prediction

train_ar = train_data['Open'].values
test_ar = test_data['Open'].values


history = [x for x in train_ar]
print(type(history))
predictions = list()
for t in range(len(test_ar)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
    # print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test_ar, predictions)
print('Testing Mean Squared Error: %.3f' % error)
error2 = smape_fun(test_ar, predictions)
print('Symmetric mean absolute percentage error: %.3f' % error2)

# view the plot for open price, prediction and actual price 

plt.figure(figsize=(12,7))
plt.plot(df['Open'], color='blue', label='Training Data')
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', label='Predicted Price')
plt.plot(test_data.index, test_data['Open'], color='red', label='Actual Price')
plt.title('HDFC Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(0,5305, 550), df['Date'][0:5305:550])
plt.legend()
plt.show()

# view the plot for predicted price and actual price 

plt.figure(figsize=(12,7))
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', 
         label='Predicted Price')
plt.plot(test_data.index, test_data['Open'], color='red', label='Actual Price')
plt.title('HDFC Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(4244,5305, 300), df['Date'][4244:5305:300])
plt.legend()
plt.show()