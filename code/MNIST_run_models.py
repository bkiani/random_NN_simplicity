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


# append data after running calculations 
def append_data():
	index_vals.append(ind)
	file_name_vals.append(model_name_str)
	dist_vals.append(steps)
	image_num_vals.append(img_num)
	type_image_vals.append(type_image)
	phi_vals.append(layer_output[0][0])
	post_act_phi_vals.append(Y_out_0[0][0])

	image_dict[type_image].append([X, X_new])

# save outputs to csv file
def save_outputs():
	pd_cols = {'name': [name]*len(file_name_vals),
		'file_name': file_name_vals,
		'index': index_vals,
		'distance': dist_vals,
		'image_num_vals': image_num_vals,
		'type_image': type_image_vals,
		'phi_val': phi_vals,
		'post_act_phi_val': post_act_phi_vals
		}

	df = pd.DataFrame(data=pd_cols)
	df.to_csv('../csv_files/'+name+'.csv')
	print(df)

	with open('../pickles/'+name+'.pickle', 'wb') as h:
		pickle.dump(image_dict, h)


model_dir = '../models/'											# directory containing saved models
saved_models = os.listdir(model_dir)

model_name_str = 'simple_standard'									# model names (prefix for saved names of models)
saved_models = [i for i in saved_models if model_name_str in i]		# list of saved models

name = 'MNIST_binary_images_2'										# name given to simulation (for saving)

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

ind = 0								# simulation number
save_i = 0							# initializing counter for saving
save_n = 1							# how many simulations before saving to csv
n_per_model = 1						# number of points to test per model

image_dict = {'train': [],
	'test': [],
	'random': []}

# loop through models
for file_i in saved_models:
	# load model
	model = load_model(model_dir+file_i)
	model_no_act = load_model(model_dir+file_i)
	# load model with no activation in final neuron
	model_no_act.layers[-1].activation = keras.activations.linear
	model_no_act = model_no_act.save('temp.h5')
	model_no_act = load_model('temp.h5')
	
	for iter_i in range(n_per_model):
		for type_image in ['train', 'test', 'random']:
			# based on type of image, get your input
			if type_image == 'train':
				img_num = np.random.choice(X_train_flat.shape[0])
				X = X_train_flat[img_num,:]
			elif type_image == 'test':
				img_num = np.random.choice(X_test_flat.shape[0])
				X = X_test_flat[img_num,:]
			elif type_image == 'random':
				X = np.random.randint(2, size= X_train_flat.shape[1])
			else:
				raise ValueError('Please specify what to do for type of image')

			X = X.reshape(-1,len(X))

			# predict both with and without final activation
			Y_out_0 = model.predict(X)
			layer_output = model_no_act.predict(X)

			# run greedy search
			steps, X_new = utils.greedy_search(model_no_act, X,
				bit_vals = [0, 1], threshold = 0., return_X = True)

			
			append_data()

		
			ind += 1
	save_i+=1
	if save_i % save_n == 0:
		save_outputs()

	utils.clear_session()
	gc.collect()


