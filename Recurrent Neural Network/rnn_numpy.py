import numpy as np
import matplotlib.pyplot as plt
plt.style('seaborn-white')

data = open('input.txt', 'r').read()

# process data and calculate indexes
chars = list(set(data))
data_size, X_size = len(data), len(chars)
print('data has %d characters, %d unique' % (data_size, X_size))

char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}

# Parameters
H_size = 100                # Size of hidden layer
T_steps = 25                # Number of time steps (length of sequence) used for training
learning_rate = 1e-1
weight_sd = 0.1             # Standard deviation of weights for initialization
z_size = H_size + X_size    # Size of concatenate (H, X) vector

# defining helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return (1-y) * y





