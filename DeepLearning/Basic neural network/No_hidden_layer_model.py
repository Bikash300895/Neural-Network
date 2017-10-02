import numpy as np
import h5py

def load_dateset():
    train_dateset = h5py.File('datasets/train_catvnoncat.h5', "r")