import utils
import pandas as pd 
import numpy as np
import time
import pickle
from math import ceil
import gc


# Need to run create_hamming_pickle.py before this to create list of 
# bit strings to search at each distance


# run search to find any bit string with different classification
# optional to set a batch_size
def batch_check(batch_size = None):
	if batch_size is None:
		Y = model.predict(X)
		print('checking')
		Y_0 = Y > out_thresh
		return np.any((not Y_base_0) == Y_0)
	else:
		n_batches = ceil(X.shape[0] / batch_size)
		for i in range(n_batches):
			if i == (n_batches - 1):
				Y = model.predict(X[batch_size*i:])
			else:
				Y = model.predict(X[batch_size*i:batch_size*(i+1)])

			Y_0 = Y > out_thresh
			print('checking '+str(i))	
			if np.any((not Y_base_0) == Y_0):
				return True
		return False


name = 'ham_validation_100_d4_try3'				# name of simulation - used to save csv file
n_models = 1000									# number of models to create and perform search on
bit_vals = [-1, 1]								# values of bits for input neurons

init = 'VarianceScaling'						# initialization for weights - see utils
out_activation = 'linear'						# type of activation in final neuron
out_thresh = 0.									# threshold to determine binary classification
n_feat = 100									# number of features in model
hidden_layers = [n_feat, n_feat]				# size of hidden layers

clear_sess_n = 1								# how many loops to run before clearing keras session
save_n = 10										# how many loops to run before saving outputs to csv

# initializing lists to store outputs of calculations
num_vals = range(n_models)
name_vals = [name]*(n_models)
num_feat_vals = []
ham_dist_vals = []
phi_vals = []
n_models_vals = [n_models]*(n_models)

# starting point of search
X_base = np.ones(n_feat)

# get bit strings at various hamming distances in a list
# run create_hamming_pickle code to create this pickle
with open('../pickles/hamming_list_100_3.pickle', 'rb') as handle:
    X_val_arr = pickle.load(handle)

X_base = X_base.reshape(-1, len(X_base))

start = time.time()

for mod_i in range(n_models):
	
	# show time taken to perform clear_sess_n loops
	# also clean up memory and clear current keras session
	if mod_i%clear_sess_n == 0:
		print(mod_i)
		print(time.time()-start)
		start = time.time()
		utils.clear_session()
		gc.collect()
	
	# create new network with randomized initializations
	model = utils.create_net(out_A_type = out_activation, 
		init_type = init,
		n_features = n_feat, 
		hidden_shape = hidden_layers)

	# predict starting point of search and determine classification
	Y_base = model.predict(X_base)
	print(Y_base)
	Y_base_0 = Y_base > out_thresh

	# run search at progressively increasing distance until point with different classification found
	dist_val = 1
	for X in X_val_arr:
		batch_size = None
		if dist_val == len(X_val_arr):
			batch_size = 10**5

		if batch_check(batch_size):
			break
		dist_val+=1
	
	# print(dist_val)

	# appending results of calculations to output to csv later
	ham_dist_vals.append(dist_val)
	num_feat_vals.append(n_feat)
	phi_vals.append(Y_base[0][0])

	if mod_i%save_n == 0:
		# placing data in dictionary and saving to csv using pandas
		pd_cols = {'Name': name_vals[:(mod_i+1)],
			'Model_Num': num_vals[:(mod_i+1)],
			'N_models': n_models_vals[:(mod_i+1)],
			'Num_features': num_feat_vals,
			'Distance_Num': ham_dist_vals,
			'Phi_Val': phi_vals
			}
		df = pd.DataFrame(data=pd_cols)
		df.to_csv('../csv_files/'+name+'.csv')
		print(df)


# placing data in dictionary and saving to csv using pandas
pd_cols = {'Name': name_vals,
	'Model_Num': num_vals,
	'N_models': n_models_vals,
	'Num_features': num_feat_vals,
	'Distance_Num': ham_dist_vals,
	'Phi_Val': phi_vals
	}
df = pd.DataFrame(data=pd_cols)
df.to_csv('../csv_files/'+name+'.csv')
print(df)

