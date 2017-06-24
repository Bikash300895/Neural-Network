# Python imports
import numpy as np

np.seterr(all='ignore')  # ignore numpy warning line multiplication of inf
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, ListedColormap
from matplotlib import cm

np.random.seed(seed=1)

"""Define and generate sample"""
nb_of_samples_per_class = 20
red_mean = [-1, 0]  # The mean of red class
blue_mean = [1, 0]  # The mean of blue class
std_dev = 1.2  # Standard deviation of both classes

# generate sample from both classes
x_red = np.random.rand(nb_of_samples_per_class, 2) * std_dev + red_mean
x_blue = np.random.rand(nb_of_samples_per_class, 2) * std_dev + blue_mean

# Merge samples in set of input variables x, and corresponding set of output variables t
X = np.vstack((x_red, x_blue))
t = np.vstack((np.zeros((nb_of_samples_per_class, 1)), np.ones((nb_of_samples_per_class, 1))))

# Plot both classes on the x1, x2 plane
plt.plot(x_red[:, 0], x_red[:, 1], 'ro', label='class red')
plt.plot(x_blue[:, 0], x_blue[:, 1], 'bo', label='class blue')
plt.grid()
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.axis([-4, 4, -4, 4])
plt.title('red vs. blue classes in the input space')
plt.show()


"""Define functions"""


# Define logistic function
def logistic(z):
    return 1 / (1 + np.exp(-z))


def nn(x, w):
    return logistic(x.dot(w.T))


def nn_predict(x, w):
    return np.around(nn(x, w))


def cost(y, t):
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))


