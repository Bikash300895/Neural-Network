""" Part 1 - Importing libraryr and dataset """
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# loading the dat (cat/nont cat)
X_train, y_train, X_test, y_test, classes = load_dataset()

# show an image 
plt.imshow(X_train[25])

"""  part 2 - Data Preprocessing """
# Changing the shape to (feature x nb_example)
X_train_flatten = X_train.reshape(-1, X_train.shape[0])
X_test_flatten = X_test.reshape(-1, X_test.shape[0])

# Standarize the data
train_set_x = X_train_flatten/255
test_set_x = X_test_flatten/255

""" part 3 - Defining NN Parameters and helper functions"""
# Helper function for initializing the weight
def initialize_paramaters(dim):
    w = np.random.randn(dim[0], 1)
    b = 0
    
    return w, b

W, b = initialize_paramaters(dim=(train_set_x.shape[0], 1))


"""  part 4 - Forword Propagation """
# Defining activation fucntions
def sigmoid(z):
    return 1/(1+np.exp(-z))

# test the tunction
#print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

def propagate(W, X, b, Y):
    # number of example 
    m = X.shape[1]
    
    # coputing gradient
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    dZ = A - Y
    dW = (np.dot(X, dZ.T))/m
    db = np.sum(dZ) / m
    
    # compute the cost
    cost = -1/m*(np.sum( (Y* np.log(A)) + ((1-Y)*np.log(1-A) )))
    
    gradients = {
                "dW" : dW,
                "db" : db
            }
    return gradients, cost

# test the forward propagation
#w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
#grads, cost = propagate(w, X, b, Y)
#print ("dw = " + str(grads["dW"]))
#print ("db = " + str(grads["db"]))
#print ("cost = " + str(cost))



"""  part 5 - Optimization """
def optimize(W, X, b, Y, num_iterations, learning_rate, print_cost = False): 
    costs = []
    
    for i in range(num_iterations):
        # forward pass
        grads, cost = propagate(W, X, b, Y)
        
        dW = grads["dW"]
        db = grads["db"]
        
        # update parameters (backprop)
        W = W - learning_rate*dW
        b = b - learning_rate*db
        
        if i %100 == 99:
            costs.append(cost)
        
        if print_cost and i%100 == 99:
            print("Cost ater iteratrions %i: %f" %(i+1, cost))
            
        params = {
                    "W": W,
                    "b": b
                }
        
    return params, grads, costs

# Testing the optimization
#params, grads, costs = optimize(w, X, b, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

#print ("w = " + str(params["W"]))
#print ("b = " + str(params["b"]))
#print ("dw = " + str(grads["dW"]))
#print ("db = " + str(grads["db"]))
        
""" Part 6 - Train and find accuracy """
def predict(W, X, b):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    
    A = sigmoid(np.dot(W.T, X) + b)
    
    for i in range(m):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
    
    return Y_prediction

# Marge all this to build the model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    W, b = initialize_paramaters(dim=(X_train.shape[0], 1))

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(W, X_train, b, Y_train,num_iterations = 2000, learning_rate = 0.5, print_cost = True)
    
    # Retrieve parameters w and b from dictionary "parameters"
    W = parameters["W"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(W, X_test, b)
    Y_prediction_train = predict(W, X_train, b)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : W, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
        
        
d = model(train_set_x, y_train, test_set_x, y_test, num_iterations = 2000, learning_rate = 0.005, print_cost = True)      
        
        