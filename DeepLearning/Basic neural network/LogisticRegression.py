""" Part 1 - Importing libraryr and dataset """
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# loading the dat (cat/nont cat)
X_train, y_train, X_test, y_test, classes = load_dataset()

# show an image 
plt.imshow(X_train[25])

"""  part 2 - Data Preprocessing """
# Changing the shape to (feature x nb_example)
X_train_flatten = X_train.reshape(-1, X_train.shape[0])
X_test_flatten = X_test.reshape(-1, X_test.shape[0])

# Standarize the data
train_set_x = X_train_flatten/255
test_set_x = X_test_flatten/255

""" part 3 - Defining NN Parameters and helper functions"""
# Helper function for initializing the weight
def initialize_paramaters(dim):
    w = np.zeros(dim)
    b = 0
    
    return w, b

W, b = initialize_paramaters(dim=(train_set_x.shape[0], 1))


"""  part 4 - Forword Propagation """

    


"""  part  - Optimization """