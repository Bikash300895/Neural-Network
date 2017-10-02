import numpy as np
import h5py

def load_dateset():
    # opening the dataset file
    train_dateset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dateset["train_set_x"][:])   # Train set features
    train_set_y_orig = np.array(train_dateset["train_set_y"][:])   # train set labels
    
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])   # Test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])   # Test set labels
    
    classes = np.array(test_dataset["list_classes"][:])
    
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

"""Part 1: importing and preprocessing data"""
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dateset()

# Show an image
import matplotlib.pyplot as plt
plt.imshow(train_set_x_orig[25])

# Reshaping data
train_x_flatten = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
test_x_flatten = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T

train_x = train_x_flatten/255
test_x = test_x_flatten/255

train_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))



""" Part 2 : Defining neural network """
def initialize_random_variable(dim):
    w = np.random.randn(dim, 1) / np.sqrt(dim)
    b = 0
    return w, b


W, b = initialize_random_variable(train_x.shape[0])


""" Part 3 : Forward Propagation """
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# definig forward propagation
def propagate(W, X, b, Y):
    Z = np.dot(W.T, X)
    A = sigmoid(Z)
    
    # number of example
    m = X.shape[1]
    
    # computing the cost
    cost = -np.sum((Y*np.log(A)) + (1-Y)*np.log(1-A))/m
    
    # derivative calculation
    dZ = A - Y
    dW = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m
    
    gradients = {
                "dW" : dW,
                "db" : db
            }
    return gradients, cost
    
gradients, cost = propagate(W, train_x, b, train_y)
    

""" Part 4 : BackProp/ Optimization """
def optimize(W, X, b, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(W, X, b, Y)
        dW = grads["dW"]
        db = grads["db"]
        
        W = W - learning_rate * dW
        b = b - learning_rate * db
        
        if i %100 == 99:
            costs.append(cost)
        
        if print_cost and i%100 == 99:
            print("Cost ater iteratrions %i: %f" %(i+1, cost))
            
        params = {
                    "W": W,
                    "b": b
                }
        
    return params, grads, costs

params, grads, costs = optimize(W, train_x, b, train_y, num_iterations=2000, learning_rate=0.01, print_cost=True)



""" Part 5 : Measuring the performance """    
def predict(W, X, b):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    
    A = sigmoid(np.dot(W.T, X) + b)
    
    for i in range(m):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
            
    return Y_prediction

def accuracy():
    Y_prediction_test = predict(W, test_x, b)
    Y_prediction_train = predict(W, train_x, b)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100))

accuracy()


# Marge all this to build the model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    W, b = initialize_random_variable(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(W, X_train, b, Y_train,num_iterations = num_iterations, learning_rate = learning_rate, print_cost = True)
    
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
    
    
d = model(train_x, train_y, test_x, test_y, num_iterations = 4000, learning_rate = 0.005, print_cost = True)  


""" Part 5 : Post implementation """
learning_rates = [0.01, 0.001, 0.0001]
accuracies = []
costs = []
for i in learning_rates:
    print("Running training for learning rate ", i)
    d = model(train_x, train_y, test_x, test_y, num_iterations = 4000, learning_rate = i, print_cost = True)      
    accuracies.append(100 - np.mean(np.abs(d["Y_prediction_test"] - test_y)) * 100)
    costs.append(d["costs"])

plt.scatter(learning_rates, accuracies)
plt.plot(learning_rates, accuracies)
plt.scatter(learning_rates, costs)    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



