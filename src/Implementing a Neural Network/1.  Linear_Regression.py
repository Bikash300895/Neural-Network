import numpy as np
import matplotlib.pyplot as plt


np.random.seed(seed=1)

# our function here is f(x) = x*2
# We are defining random data for that function
x = np.random.uniform(0,1,20)

def f(x):
    return x * 2

noise_variance = 0.2
noise = np.random.randn(x.shape[0]) * noise_variance

t = f(x) + noise

plt.plot(x, t, 'o', label='t')

# plot initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')

plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.grid()
plt.legend(loc=2)

# define the neural network function 
def nn(x, w):
    return x * w

def cost(y, t):
    return ((t - y)**2).sum()

# define the weight values
ws = np.linspace(0, 4, num=100)
cost_ws = np.vectorize(lambda w: cost(nn(x, w), t))(ws)

# plot the weight vs cost
plt.plot(ws, cost_ws, 'r-')

def gradient(w, x, t):
    return 2 * x * (nn(w, x) - t)

def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient(w_k, x, t).sum()

# set the initial weight parameter
w = 0.1
# set learning rate
learning_rate = 0.1

nb_of_iterations = 4
w_cost = [(w, cost(nn(x, w), t))]


for i in range(nb_of_iterations):
    dw = delta_w(w, x, t, learning_rate)
    w = w - dw
    w_cost.append((w, cost(nn(x, w), t)))

for i in range(len(w_cost)):
    print('w({}): {:.4f} \t cost: {:.4f}'.format(i, w_cost[i][0], w_cost[i][1]))












