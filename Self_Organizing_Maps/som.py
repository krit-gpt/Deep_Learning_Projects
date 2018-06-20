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

'''
MiniSom 
x,y = dimensions of the SOM, upto us, should not be small, because we want the outliers to be present prominently.
Depends on the input customers. We have less customers, so 10 x 10 grid will do.

input_len == No of inputs in the input data, dimensions of the input data == 15!

sigma == radius of the neighbours.

learning_rate - decides by how much the weights are updated, during each iteration.
the higher the rate, the faster is the convergance.

'''
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# give random weights
som.random_weights_init(X)

# train the SOM, the number of iterations to train the data.
som.train_random(data = X, num_iteration = 100)


# Visualizing the results = from scratch.
from pylab import bone, pcolor, colorbar, plot, show

# bone() - window that will contain the map, white window without anything in it.
bone()

# pcolor() -- put the different winngin nodes on to the map, 
# different colors for the different range values for the different inter-neuron distances
# som.distance_map() gives all the different mean distances for all the winning nodes in a matrix, take its transpose -- .T
# All colors correpsonding to the Mean - interneuron distances.
pcolor(som.distance_map().T)

#colorbar() - legend of all the colors, to tell what each color represents
# Highest mean inter-neuron distances correspond to the white color, lowest -- black color.
# Defaulters -- far from the general, because outliers.
# winning nodes which have large MIDs -- outliers --frauds -- white nodes on the map
colorbar()


#markers -- to tell if the customers associated with each node fraud or not.
markers = ['o', 's']

# red - fraud, green - non fraud
colors = ['r', 'g']


for i, x in enumerate(X):
	#for each customer, get its winning node, x - customer, w - winning node.
    w = som.winner(x)
    # now place the colored marker for the winning node
    # coordinates of the marker, palce the marker at the center of the square (square -- each node)
    # w[0], w[1] -- coordinates of the winning node, the lower left corner of the square.
    # y -- gives the output, from earlier! so use y[i] -- tells if fraud or not; 
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
    # markeredgecolor -- marks the edge of the color
    # markerfacecolor -- color the inside of the marker
    # markersize -- how big the marker has to be
    # markeredgewidth -- width of the marker.


show()

# Finding the frauds
# Find the customers associated with the fraud nodes as represented by the SOM.
# No inverse mapping function, so get all the mappings between the winning nodes to the customers.
# mappings is a dictionary of all the nodes --> list of the customers associated with it, and the number of customers assoicated with the node.

mappings = som.win_map(X)

# (8,1) and (6,8) are the coordinates of the WHITE nodes -- got that after seeing the map
# now, get a combined list of all the customers associated with them, concatenate along teh vertical axis, so axis = 0
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)

#reverse the transformation to get the original data.
frauds = sc.inverse_transform(frauds)















