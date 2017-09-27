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
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward


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
        


























