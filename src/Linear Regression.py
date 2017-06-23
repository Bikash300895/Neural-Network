"""Importing library"""
import numpy
import matplotlib.pyplot as plt

"""Preparing the data"""
# Define the vector of input samples as x, with 20 values sampled from a uniform distribution
# between 0 and 1
x = numpy.random.uniform(0, 1, 20)


# Generate the target values t from x with small gaussian noise so the estimation won't
# be perfect.
# Define a function f that represents the line that generates t without noise
def f(x):
    return x * 2


# Create the targets t with some gaussian noise
noise_variance = 0.2  # Variance of the gaussian noise
# Gaussian noise error for each sample in x
noise = numpy.random.randn(x.shape[0]) * noise_variance
# Create targets t
t = f(x) + noise

# Plotting the data into graph
plt.scatter(x, t)

"""Implementing neural network functions"""


# define the neural network
def nn(x, w):
    return x * w


# define a cost function
def cost(y, t):
    return ((t - y) ** 2).sum()

"""plot the cost vs the given weight"""
ws = numpy.linspace(0, 4, num=100)
cost_ws = numpy.vectorize(lambda w: cost(nn(x, w), t)) (ws) # cost for each weight in ws

# plot it
plt.plot(ws, cost_ws)
plt.grid()

