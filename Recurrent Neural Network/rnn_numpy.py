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

## Parameters
H_size = 100                # Size of hidden layer
T_steps = 25                # Number of time steps (length of sequence) used for training
learning_rate = 1e-1
weight_sd = 0.1             # Standard deviation of weights for initialization
z_size = H_size + X_size    # Size of concatenate (H, X) vector

## defining helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return (1-y) * y


## Initialize the parameters
W_f = np.random.randn(H_size, z_size) * weight_sd + 5.0
b_f = np.random.randn((H_size, 1))

W_i = np.random.randn(H_size, z_size) * weight_sd + 0.5
b_i = np.zeros((H_size, 1))

W_C = np.random.randn(H_size, z_size) * weight_sd
b_C = np.zeros((H_size, 1))

W_o = np.random.randn(H_size, z_size) * weight_sd + 0.5
b_o = np.zeros((H_size, 1))

#For final layer to predict the next character
W_y = np.random.randn(X_size, H_size) * weight_sd
b_y = np.zeros((X_size, 1))


## Gradients
dW_f = np.zeros_like(W_f)
dW_i = np.zeros_like(W_i)
dW_C = np.zeros_like(W_C)

dW_o = np.zeros_like(W_o)
dW_y = np.zeros_like(W_y)

db_f = np.zeros_like(b_f)
db_i = np.zeros_like(b_i)
db_C = np.zeros_like(b_C)

db_o = np.zeros_like(b_o)
db_y = np.zeros_like(b_y)









