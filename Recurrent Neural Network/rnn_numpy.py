import numpy as np
import matplotlib.pyplot as plt
plt.style('seaborn-white')

data = open('input.txt', 'r').read()

# process data and calculate indexes
chars = list(set(data))
data_size, X_size = len(data), len(chars)
print('data has %d characters, %d unique' % (data_size, X_size))

# Parameters
H_size = 100            # Size of hidden layer
T_steps = 25            # Number of time steps (length of sequence) used for training
learning_rate = 1e-1
weight_sd = 0.1         # Standard deviation of weights for initialization