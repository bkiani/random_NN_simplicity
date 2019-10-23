import utils
import pandas as pd 
import numpy as np
import time
import pickle
from math import ceil
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
import gc


# option to check in batches if inputs are very large
def batch_check(model, X, Y_base_0, batch_size = None, out_thresh = 0, n_features = 1000):
	if batch_size is None: 
		# if no batch size given, then just predict on whole dataset
		Y = model.predict(X)

		# check if any differently classified
		if np.any((not Y_base_0) == Y_0):
			return np.where((not Y_base_0) == Y_0)[0][0]+1
		else:
			return Num_features													# if no differently classified, then return the number of features (num bits to flip)
	
	else:
		# running check in batches
		n_batches = ceil(n_features / batch_size)
		for i in range(n_batches):
			if i == (n_batches - 1):
				Y = model.predict(X[batch_size*i:])
				Y_0 = Y > out_thresh
			else:
				Y = model.predict(X[batch_size*i:batch_size*(i+1)])
				Y_0 = Y > out_thresh
			if np.any((not Y_base_0) == Y_0):
				return batch_size*i + np.where((not Y_base_0) == Y_0)[0][0]+1
		return n_features


# get starting point of search
# IMPORTANT: set n here to number of models input value below !!!!
def return_X(id_i, n = 1000):
	a = X_inputs[int(id_i/n)]
	return a

# run model one time and find number of bit flips until classifing an input X differently
def run_model(id_i):

	X = return_X(id_i)											# get starting bit string

	init = 'VarianceScaling'									# variance of weights set by number of fan in 
	out_activation = 'linear'									# no activation i nfinal neuron
	out_thresh = 0.												# final neuron values above or below threshold to determine classification
	n_feat = X.shape[1]											# number of features 
	hidden_layers = [n_feat, n_feat]							# shape of hidden layers


	# creating network
	model = utils.create_net(out_A_type = out_activation, 
		init_type = init,
		n_features = n_feat, 
		hidden_shape = hidden_layers)

	# get classification of starting bit string
	X_base = np.ones(n_feat).reshape(-1,n_feat)
	Y_base = model.predict(X_base)
	Y_base_0 = Y_base > out_thresh

	# run check to see after how many flips we get a different classification
	steps = batch_check(model, X, Y_base_0, 
		batch_size =500, 
		out_thresh = out_thresh,
		n_features = n_feat)

	# clear Keras session (remove junk)
	utils.clear_session()
	gc.collect()

	return [steps, Y_base[0][0]]


# saves results to a csv file
def save_results(results):
	results_out = list(zip(*results))

	pd_cols = {'Name': name_vals[:len(results_out[0])],
		'Model_Num': num_vals[:len(results_out[0])],
		'N_models': n_models_vals[:len(results_out[0])],
		'Num_features': num_feat_vals[:len(results_out[0])],
		'Distance_Num': results_out[0],
		'Phi_Val': results_out[1]
		}

	df = pd.DataFrame(data=pd_cols)
	df.to_csv('../csv_files/'+name+'.csv')
	print(df)
	return df

n_models = 1000														# number of models to run search on for each feature size
n_feat_arr = [100, 200, 500]										# feature sizes to run search on

X_inputs = []														# list containing starting bit strings

num_feat_vals = []													# initializing storage variables for outputs
for n_feat in n_feat_arr:
	num_feat_vals.extend([n_feat]*n_models)
	X_inputs.append(utils.direct_search_X(n_feat))

if __name__ == '__main__':
	name = 'direct_search_100_200_500'
	bit_vals = [-1, 1]

	num_vals = range(n_models*len(n_feat_arr))						# list cotaining number of features (for csv output)
	name_vals = [name]*(n_models*len(n_feat_arr))					# list containing name of simulation (for csv output)
	ham_dist_vals = []												# list containing hamming distances (for csv output)	
	phi_vals = []													# list containing output neuron values (for csv output)
	n_models_vals = [n_models]*(n_models*len(n_feat_arr))			# list contaning number of models tested (for csv output)

	# initializing storage variables for outputs
	results_out = [[],[]]
	results = []

	# loop through feature sizes
	for feat_i in range(len(n_feat_arr)):
		
		# loop through number of models to create for each size
		for i in range(n_models):
			
			a = run_model(feat_i*n_models+i)
			results.append(a)

			# save results after every 10 iterations
			if i%10 == 0:
				df = save_results(results)

		df = save_results(results)
			


