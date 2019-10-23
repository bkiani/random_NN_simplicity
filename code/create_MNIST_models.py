import MNIST_utils
import utils
import numpy as np
from keras.datasets import mnist
import gc
import pandas as pd

# trains a number of neural nets to run on MNIST data
# classifies whether digit is even or odd
# models are saved to the ../models directory for later analysis



# creates neural net and trains it
def run_model():

	# model initialization parameters
	init = 'VarianceScaling'
	out_activation = 'sigmoid'								# using sigmoid activation at output for binary classification
	out_thresh = 0.											# threshold of output for non-activated final neuron

	# shape of hidden layers
	hidden_layers = [n_feat, n_feat]

	utils.clear_session()
	gc.collect()

	model = utils.create_net(out_A_type = out_activation, 
		init_type = init,
		n_features = n_feat, 
		hidden_shape = hidden_layers)

	model.compile(optimizer = 'adam', 
		loss = 'binary_crossentropy', 
		metrics = ['accuracy'])

	model.fit(X_train_flat ,y_train_even, epochs = 20, 
		verbose = 2)

	return model

name = 'simple_standard'									# name of models for prefix

(X_train, y_train), (X_test, y_test) = mnist.load_data()	

binary_thresh = 25											# threshold to convert pixel values to binary
X_train = MNIST_utils.convert_to_binary(
	X_train, binary_thresh)
X_test = MNIST_utils.convert_to_binary(
	X_test, binary_thresh)

n_test = 2000												# number of models to create
np.set_printoptions(threshold=np.inf)

# determine if digit is even in train and test set
y_train_even = (y_train % 2 == 0).astype(int)				
y_test_even = (y_test % 2 == 0).astype(int)

# flatten image to 1d vector
X_train_flat = MNIST_utils.flatten_img(X_train)
X_test_flat = MNIST_utils.flatten_img(X_test)
n_feat = X_train_flat.shape[1]


# run and save models
for i in range(n_test):
	print(i)
	model_i = run_model()
	y_pred = model_i.predict(X_test_flat)
	print(MNIST_utils.accuracy_from_prob(y_pred, y_test_even))
	# model_i.save('../models/'+name+'_{:05d}.h5'.format(i))
