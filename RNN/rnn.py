# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# extracting the Open Google stock price, and since the model needs an input a 2D array, converting the
# model to a 2D array.
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
# Normalization applied. 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) #normalize to a rang eod 0-1

# fir the normalization and transform each input entry to a range of 0-1 
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping

# Reshaping is done in order to add a dimension of timestep in the training set, bcs LSTM hai na.
# time step is needed.
'''
Look at documentation -- might have changed..
np.reshape( number of observations in the training set,  ( number of features in the input ), )
'''

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#//////////////////////////////////////////BUILD RNN//////////////////////////////////////////////////// 

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
# Making a regression model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
# making a regression model, hence, called it regressor
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
#Adding the LSTM layer, The number of units
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
# Doing regression, hence loss is the mean squared error.
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)

#inverse transform the results to convert from a scale of [0,1] to original stock prices.
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error

# rmse between y (true) and y hat(our prediction)
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

#showing rmse as a percentage of the original value
# 800 is the average values of the google stock price, the values are around 800.
print(rmse/800)






