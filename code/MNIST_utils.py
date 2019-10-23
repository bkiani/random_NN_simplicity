import numpy as np
import statistics
import pandas as pd
import pickle
import time
import h5py
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D


# calculate hamming distance between two vectors
def hamming_dist_calc(vec1, vec2):
	return np.count_nonzero( vec1 != vec2 )

# convert image pixels to binary
def convert_to_binary(arr_in, threshold):
	return (arr_in>threshold).astype(int)

# flatten image into 1d vector
def flatten_img(X_in):
	X_shape = X_in.shape
	X_out = X_in.reshape(X_shape[0], X_shape[1]*X_shape[2])
	return X_out

# get accuracy of model
def accuracy_from_prob(pred, actual, thresh=0.5):
	pred = (pred > thresh).astype(int).flatten()

	n = len(pred)
	n_wrong = np.count_nonzero(pred != actual)
	print(np.count_nonzero(pred[:10] != actual[:10]))
	return (n-n_wrong)/n