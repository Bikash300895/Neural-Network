import copy, numpy as np

np.random.seed(0)


# Compute sigmoid non linearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# Convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


