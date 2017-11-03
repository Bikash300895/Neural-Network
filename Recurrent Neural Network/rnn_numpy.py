import numpy as np
import matplotlib.pyplot as plt
plt.style('seaborn-white')

data = open('input.txt', 'r').read()

# process data and calculate indexes
chars = list(set(data))
data_size, X_size = len(data), len(chars)
print('data has %d characters, %d unique' % (data_size, X_size))

