""" Part 1 - Importing libraries and datasets """
# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
np.random.seed(1) # set a seed so that the results are consistent

""" Part 2 - Data preprocessing """
# Loading the data
X, Y = load_planar_dataset()

# viasualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)

""" Part 3 - Defining NN and helper function """
def layer_size(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

""" Part 4 - Forword propagation """
def forward_propagation(X, parameters):
    # Retriving the parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Forword propagation calcualtion
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

# Calculating Cost
def compute_cost(A2, Y, parameters):
    m = Y.shape[1] # number of example
    cost = -1/m*(np.sum( (Y* np.log(A2)) + ((1-Y)*np.log(1-A2) )))
    
    return cost


""" Part 5 - Back Propagetaion """
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = 1/m*(np.sum(dZ2, axis=1, keepdims=True))
    dZ1 = np.dot(W2.T, dZ2) *  (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = 1/m*(np.sum(dZ1, axis=1, keepdims=True))

    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads



