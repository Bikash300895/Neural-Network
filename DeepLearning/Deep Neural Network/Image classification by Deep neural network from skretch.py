""""Part 1 - Importing libaries"""
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *
from testCases_v3 import *
from dnn_utils_v2 import sigmoid_backward,  relu_backward


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)


"""Part 2 - Importing and preprocessing data"""
# loading dataset
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Reshape trainning and test example
train_x_flat = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flat = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standarize the data to have values between 0 and 1
train_x = train_x_flat/255
test_x = test_x_flat/255


"""Part 3 - Initialize the neural network"""
np.random.seed(3)
def initialize_parameters_deep(layer_dims):
    L = len(layer_dims)
    parameters = {}
    
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters
        

""" Part 4 - Forword propagation """
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    
    return Z, linear_cache


def sigmoid(z):
    A = 1/(1+np.exp(-z))
    cache = z
    
    return A, cache

def relu(z):
    A = np.maximum(0, z)
    cache = z
    
    return A, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation=="sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forword(X, parameters):
    caches = []
    L = len(parameters)//2 + 1
    A = X
    
    for l in range(1, L-1):
        A, cache = linear_activation_forward(A, parameters["W"+str(l), parameters["b"+str(l)], "relu"])
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters["W"+str(L-1), parameters["b"+str(L-1)], "sigmoid"])
    caches.append(cache)    
     
    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "sigmoid"
    























