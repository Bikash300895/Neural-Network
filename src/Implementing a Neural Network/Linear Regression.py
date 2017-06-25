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


"""Defining gradient descent"""
def gradient(w, x, t):
    return 2 * x * (nn(x, w) - t)


def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient(w_k, x, t).sum()

# Set the initial weight parameter
w = 0.1
# set learning rate
learning_rate = 0.1

# start performing the gradient updates, and print the weight and cost:
nb_of_iterations = 10
w_cost = [(w, cost(nn(x, w), t) )]
for i in range(nb_of_iterations):
    dw = delta_w(w, x, t, learning_rate) # Get the delta w update
    w = w - dw # update the current weight parameter
    w_cost.append((w, cost(nn(x, w), t)))
    
    
# Plot the first 2 gradient descent updates
plt.plot(ws, cost_ws, 'r-')  # Plot the error curve


 # Plot the fitted line agains the target line
# Plot the target t versus the input x
plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
# plot the fitted line
plt.plot([0, 1], [0*w, 1*w], 'r-', label='fitted line')
plt.xlabel('input x')
plt.ylabel('target t')
plt.ylim([0,2])
plt.title('input vs. target')
plt.grid()
plt.legend(loc=2)
plt.show()
