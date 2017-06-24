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

# input variable
alpha = 0.1         # Learning rate
input_dim = 2       # we are adding two input
hidden_dim = 16
output_dim = 1


# initialize newral network weights
synapse_0 = 2 * np.random.random((input_dim, hidden_dim))
synapse_1 = 2 * np.random.random((hidden_dim, output_dim))
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim))

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

