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
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))


# Plot the cost function of the weights
nb_of_ws = 100
ws1 = np.linspace(-5, 5, num=nb_of_ws)
ws2 = np.linspace(-5, 5, num=nb_of_ws)
ws_x, ws_y = np.meshgrid(ws1, ws2)
cost_ws = np.zeros((nb_of_ws, nb_of_ws))

# fill the cost matrix for each combination of weights
for i in range(nb_of_ws):
    for j in range(nb_of_ws):
        cost_ws[i, j] = cost(nn(X, np.asmatrix([ws_x[i, j], ws_y[i, j]])), t)

# Plot the cost function surface
plt.contourf(ws_x, ws_y, cost_ws, 20, cmap=cm.pink)
cbar = plt.colorbar()
cbar.ax.set_ylabel('$\\xi$', fontsize=15)
plt.xlabel('$w_1$', fontsize=15)
plt.ylabel('$w_2$', fontsize=15)
plt.title('Cost function surface')
plt.grid()
plt.show()


# Define the gradient descent
def gradient(w, x, t):
    return (nn(x, w) -t).T * x


# define the update function delta w which return the delta w for each weight in a vector
def delta_w(w_k, x, t ,learning_rate):
    return learning_rate * gradient(w_k, x ,t)


