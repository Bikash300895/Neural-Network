# python import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm       # ColorMaps

np.random.seed(seed=1)


# Define the dataset
number_of_samples_per_class = 20
blue_mean = [0]
red_left_mean = [-2]
red_right_mean = [2]

std_dev = 0.5

# generate sample from both classes
x_blue = np.random.randn(number_of_samples_per_class, 1) * std_dev + blue_mean
x_red_left = np.random.randn(int(number_of_samples_per_class/2), 1) * std_dev + red_left_mean
x_red_right = np.random.randn(int(number_of_samples_per_class/2), 1) * std_dev + red_right_mean

# Marge samples in set of input variables x, and corresponding set of output variables t
x = np.vstack((x_blue, x_red_left, x_red_right))
t = np.vstack((np.ones((x_blue.shape[0],1)), 
               np.zeros((x_red_left.shape[0],1)), 
               np.zeros((x_red_right.shape[0], 1))))

# Plot the samples from both classes as lines on a 1D space
plt.figure(figsize=(8, 0.5))
plt.xlim(-3, 3)
plt.ylim(-1,1)
# plot samples
plt.plot(x_blue, np.zeros_like(x_blue), 'b|', ms=30)
plt.plot(x_red_left, np.zeros_like(x_red_left), 'r|', ms = 30) 
plt.plot(x_red_right, np.zeros_like(x_red_right), 'r|', ms = 30) 
plt.gca().axes.get_yaxis().set_visible(False)
plt.title('Input samples from the blue and red class')
plt.xlabel('$x$', fontsize=15)
plt.show()