import copy, numpy as np

np.random.seed(0)


# Compute sigmoid non linearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# Convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

