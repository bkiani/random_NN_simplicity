import utils
import pandas as pd 
import numpy as np
import time
import pickle
import multiprocessing as mp
import gc
import os


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
name = 'greedy_search_AWS_p_10000_0'

# run greedy search
def run_model(n_feat, multi_gpu = None):
	import utils

	init = 'VarianceScaling'
	out_activation = 'linear'
	out_thresh = 0.

	hidden_layers = [n_feat, n_feat]

	utils.clear_session()
	gc.collect()

	model = utils.create_net(out_A_type = out_activation, 
		init_type = init,
		n_features = n_feat, 
		hidden_shape = hidden_layers)

	X_base = np.ones(n_feat).reshape(-1,n_feat)
	Y_base = model.predict(X_base)
	# print('predicted')

	steps = utils.greedy_search(model, X_base)
	# print('finished search')

	return [steps, Y_base[0][0]]

n_models = 15											# number of models to test per feature size
n_feat_arr = [100, 250, 500, 1000, 2000, 2500, 10000] 	# features to run search on

# placeholders for data to be outputted to csv
num_feat_vals = []
X_inputs = []
for n_feat in n_feat_arr:
	num_feat_vals.extend([n_feat]*n_models)

if __name__ == '__main__':

	bit_vals = [-1, 1]

	# formatting data for later output to csv
	num_vals = range(n_models*len(n_feat_arr))				# number of model for feature size
	name_vals = [name]*(n_models*len(n_feat_arr))			# name of simulation
	ham_dist_vals = []										# output of hamming distance calculation
	phi_vals = []											# output neuron value
	n_models_vals = [n_models]*(n_models*len(n_feat_arr))	# number of models simulated for feature size

	results = []
	start = time.time()
	prev_i = num_feat_vals[0]
	time_count = 0
	save_count = 1
	
	for i in num_feat_vals:
		if time_count == 1:
			print(time.time()-start)
			start = time.time()
			time_count = 0
		else:
			time_count +=1

		# run model with greedy search and append results
		results.append(run_model(i))

		if save_count != 5:
			save_count += 1
		else:
			save_count =1
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


