import os
from keras.models import load_model
import MNIST_utils
from keras.datasets import mnist
import numpy as np
import utils
from keras import backend as K
import keras
import pandas as pd
import gc
import pickle
from math import ceil


def batch_check(model, X, Y_base_0, batch_size = None, out_thresh = 0., n_features = 1000):
	if batch_size is None: 
		# if no batch size given, then just predict on whole dataset
		Y = model.predict(X)

		# check if any differently classified
		if np.any((not Y_base_0) == Y_0):
			return np.where((not Y_base_0) == Y_0)[0][0]+1
		else:
			return n_features													# if no differently classified, then return the number of features (num bits to flip)
	
	else:
		# running check in batches
		n_batches = ceil(n_features / batch_size)
		for i in range(n_batches):
			if i == (n_batches - 1):
				Y = model.predict(X[batch_size*i:])
				print(Y)
				Y_0 = Y > out_thresh
			else:
				Y = model.predict(X[batch_size*i:batch_size*(i+1)])
				Y_0 = Y > out_thresh
			if np.any((not Y_base_0) == Y_0):
				return batch_size*i + np.where((not Y_base_0) == Y_0)[0][0]+1
		return n_features


# append data after running calculations 
def append_data():
	index_vals.append(ind)
	file_name_vals.append(file_i)
	dist_vals.append(steps)
	image_num_vals.append(img_num)
	type_image_vals.append(type_image)
	phi_vals.append(layer_output[0][0])
	post_act_phi_vals.append(Y_out_0[0][0])
	test_accuracy_vals.append(test_accuracy)

	# image_dict[type_image].append([X, X_new])

# save outputs to csv file
def save_outputs():
	pd_cols = {'name': [name]*len(file_name_vals),
		'file_name': file_name_vals,
		'index': index_vals,
		'distance': dist_vals,
		'image_num_vals': image_num_vals,
		'type_image': type_image_vals,
		'phi_val': phi_vals,
		'post_act_phi_val': post_act_phi_vals,
		'test_accuracy': test_accuracy_vals
		}

	df = pd.DataFrame(data=pd_cols)
	df.to_csv('../csv_files/'+name+'.csv')
	print(df)

	# with open('../pickles/'+name+'.pickle', 'wb') as h:
	# 	pickle.dump(image_dict, h)


model_dir = '../models/'											# directory containing saved models
saved_models = os.listdir(model_dir)

model_name_str = 'simple_standard'									# model names (prefix for saved names of models)
saved_models = [i for i in saved_models if model_name_str in i]		# list of saved models

name = 'MNIST_false_correct_averaged'										# name given to simulation (for saving)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# threshold for convertin MNIST data to binary format
binary_thresh = 25
X_train = MNIST_utils.convert_to_binary(
	X_train, binary_thresh)
X_test = MNIST_utils.convert_to_binary(
	X_test, binary_thresh)

# determine if digit is even (convert to binary classification)
y_train_even = (y_train % 2 == 0).astype(int)
y_test_even = (y_test % 2 == 0).astype(int)

# convert images to 1d vectors
X_train_flat = MNIST_utils.flatten_img(X_train)
X_test_flat = MNIST_utils.flatten_img(X_test)
n_feat = X_train_flat.shape[1]


# initializing lists for saving outputs of calculations
index_vals = []
file_name_vals = []
dist_vals = []
image_num_vals = []
type_image_vals = []
phi_vals = []
post_act_phi_vals = []
test_accuracy_vals = []

ind = 0								# simulation number
save_i = 0							# initializing counter for saving
save_n = 1							# how many simulations before saving to csv
n_per_model = 100					# number of points to test per model

image_dict = {'train': [],
	'test': [],
	'random': []}

# loop through models
for file_i in saved_models:
	print(file_i)
	# load model
	if file_i[-5:-3] != '00' and file_i[-5:-3] != '09':
		print('skipping')
		continue
	model = load_model(model_dir+file_i)
	y_pred = model.predict(X_test_flat)
	test_accuracy = MNIST_utils.accuracy_from_prob(y_pred, y_test_even)
	print(y_test_even)
	print(y_pred)
	correct_inds_test = np.where(np.sign(y_pred.reshape(-1) - 0.5) == np.sign( y_test_even.astype(np.float32) - 0.5 ))[0]
	false_inds_test = np.where(np.sign(y_pred.reshape(-1) - 0.5) != np.sign( y_test_even.astype(np.float32) - 0.5))[0]

	y_pred = model.predict(X_train_flat)

	correct_inds_train = np.where(np.sign(y_pred.reshape(-1) - 0.5) == np.sign( y_train_even.astype(np.float32) - 0.5 ))[0]
	print(correct_inds_train)
	false_inds_train = np.where(np.sign(y_pred.reshape(-1) - 0.5) != np.sign( y_train_even.astype(np.float32) - 0.5))[0]


	model_no_act = load_model(model_dir+file_i)
	# load model with no activation in final neuron
	model_no_act.layers[-1].activation = keras.activations.linear
	model_no_act = model_no_act.save('temp.h5')
	model_no_act = load_model('temp.h5')
	
	for iter_i in range(n_per_model):
		for type_image in ['train_correct', 'train_false', 'test_correct', 'test_false']:
			# based on type of image, get your input
			if type_image == 'train_correct':
				img_num = np.random.choice(correct_inds_train)
				X = X_train_flat[img_num,:]
			elif type_image == 'train_false':
				img_num = np.random.choice(false_inds_train)
				X = X_train_flat[img_num,:]
			elif type_image == 'test_correct':
				img_num = np.random.choice(correct_inds_test)
				X = X_test_flat[img_num,:]
			elif type_image == 'test_false':
				img_num = np.random.choice(false_inds_test)
				X = X_test_flat[img_num,:]
			elif type_image == 'random':
				X = np.random.randint(2, size= X_train_flat.shape[1])
			else:
				print(type_image)
				raise ValueError('Please specify what to do for type of image')

			X = X.reshape(-1,len(X))

			# predict both with and without final activation
			Y_out_0 = model.predict(X)
			layer_output = model_no_act.predict(X)

			# run greedy search
			# steps, X_new = utils.greedy_search(model_no_act, X,
			# 	bit_vals = [0, 1], threshold = 0., return_X = True)
			steps = utils.direct_search_batch(model_no_act, 
					utils.direct_search_X_random(X), 
					layer_output > 0., 
					batch_size =500, 
					out_thresh = 0.,
					n_features = 784)

			
			append_data()

		
			ind += 1
	save_i+=1
	if save_i % save_n == 0:
		save_outputs()

	utils.clear_session()
	gc.collect()


