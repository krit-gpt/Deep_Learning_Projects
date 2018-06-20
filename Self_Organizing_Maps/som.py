# Self Organizing Map
# Return a list of potential fradulent customers using unsupervised Learning. 
# Dataset from the UCI Machine Learning Library.
# Customer Segmentation, in order to get the segment representing the potential fraud.
'''
Customers are the input to the SOM and they are going to be represented into an output space.
Neurons have weights which are dimensionally equal to the number of dimension of the customer, so the number of
weights in each neuron is equal to the number of input dimensions of teh customer.

So, basically, the input points are going to be mapped to a new output space.
For each customer, the output will be a neuron that is closest to the customer. This neuron is called the Winning node.

Neighbourhood function like the Gaussian Neighbourhood function, to update the weights of the neighbours of the winning node, and move them closer to the 
winning node. Do this for all the customers in the data and do many times for the data. Each time we repeat it, the output space decreases, 
and loses dimensions. Finally it reaches a point where the neighbourhood stops decreasing / output space stops decreasing.

Now, fraudsters will be outliers in the data. To identify the outliers, identify the outlier neurons in the outsput space.
To identify it, we need the mean interneuron distance (MID), for each neuron calculate the Euclidean distance between this neuron and 
all its neighbours. Define the neughbourhood manually.
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')

# X -- input data, [rows, columns] -- [all_rows, all_columns except the last one] -- [:, :-1]
X = dataset.iloc[:, :-1].values

# Y -- to test the data!!! NOT TO TRAIN, because unsupervised technique.
# [all_rows, last column only]
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
# fit and then transform X into a range of (0,1)


# Training the SOM
# Minisom is the package for SOMs
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)