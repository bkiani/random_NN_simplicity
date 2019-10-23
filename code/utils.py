from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
from keras import callbacks
import numpy as np
import itertools
# from lempel_ziv_complexity import lempel_ziv_complexity
from keras.initializers import glorot_uniform
from keras.initializers import VarianceScaling
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from scipy.stats import kstat
import pandas as pd
from math import ceil
from random import shuffle


# create random neural network with intializations as specified by the various parameters
def create_net(n_features = 7,  
	hidden_shape = [40,40], 
	n_outputs = 1,
	A_type = 'relu',
	out_A_type = None,
	init_type = 'glorot_uniform',
	b_init_type = 'zeros',
	include_bias = False,
	var_scale_val = 2.0):
	
	model = Sequential()

	net_shape = [n_features]+hidden_shape

	# variance scaling where weights are set based on number of fan_in neurons
	if init_type == 'VarianceScaling':
		init_type = VarianceScaling(var_scale_val,
			mode = 'fan_in', distribution = 'normal')

	# adding hidden layers
	for layer_i in range(len(net_shape)-1):
		model.add(Dense(net_shape[layer_i+1],
			input_shape = (net_shape[layer_i],),
			activation = A_type,
			use_bias = include_bias,
			kernel_initializer=init_type,
			bias_initializer=b_init_type))

	# adding output layer
	model.add(Dense(n_outputs,
		activation = out_A_type,
		use_bias = include_bias))

	return model


# reinitializes weights without creating new network
# it seems there is negligible improvement with this method
def reinitialize_weights(model, 
	init_type = 'glorot_uniform',
	var_scale_val = 2.0):
	sess = K.get_session()
	if init_type == 'glorot_uniform':
		initializer = glorot_uniform
	elif init_type == 'VarianceScaling':
		initializer = VarianceScaling(var_scale_val,
			mode = 'fan_in', distribution = 'normal')

	initial_weights = model.get_weights()
	new_weights = [initializer()(w.shape).eval(session = sess) for w in initial_weights]
	model.set_weights(new_weights)
	return model


# train a model using stochastic gradient descent
def train_SGD(model, x_train, y_train,
	n_epochs = 10,
	n_batch = 128, 
	learn_rate = 0.01,
	momentum_val=0.0, 
	decay_val=0.0, 
	nesterov_val=False,
	verbose_val = 0, 
	early_stop_val = None):
	
	sgd = SGD(lr = learn_rate, 
		momentum=momentum_val, 
		decay=decay_val, 
		nesterov=nesterov_val)
	model.compile(loss='binary_crossentropy',
		optimizer=sgd,
		metrics=['accuracy'])

	# customized addition to Keras that stops SGD if a loss val is achieved
	if early_stop_val is not None:
		callback = [callbacks.EarlyStoppingByLossVal(
			monitor='val_loss', value= early_stop_val, verbose=1)]
		model.fit(x_train, y_train,
	          epochs=n_epochs,
	          batch_size=n_batch,
	          verbose = verbose_val,
	          callbacks = callback,
	          validation_data=(x_train, y_train))
	else:
		model.fit(x_train, y_train,
	          epochs=n_epochs,
	          batch_size=n_batch,
	          verbose = verbose_val)

	if early_stop_val is None:
		return model
	else:
		return model, callback


# creates a list of all bit strings of length n_bits
def get_bit_array(n_bits = 7, replace_vals = None):
	bin_enum = list(itertools.product([0, 1], 
		repeat=n_bits))
	np_out = np.asarray(bin_enum)
	if replace_vals is not None:
		np_out[np_out == 0] = replace_vals[0]
		np_out[np_out == 1] = replace_vals[1]
	return np_out


# creates a list of random bit strings of length n_bits
def get_sample_bit_array(n_bits = 7, n_samples = 1000, 
	replace_vals = None):
	np_out = np.random.randint(2, size=(n_samples, n_bits))
	if replace_vals is not None:
		np_out[np_out == 0] = replace_vals[0]
		np_out[np_out == 1] = replace_vals[1]
	return np_out


def flatten_outputs(outputs):
	Y_0 = outputs.astype('int').astype('str').tolist()
	Y_0 = [item for sublist in Y_0 for item in sublist]
	return ''.join(Y_0)

def get_lz_comp(str_flat):
	return lempel_ziv_complexity(str_flat)


def clear_session():
	K.clear_session()

def get_train_test_set(X, y, 
	test_p = 0.9):
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_p)
	return X_train, X_test, y_train, y_test



def add_fields_dictionary(dict_in):
	Y_vals = dict_in['Y_values']
	ave_Ys = [np.mean(i) for i in Y_vals]
	var_Ys = [np.var(i) for i in Y_vals]
	dict_in['Ave_Y'] = ave_Ys
	dict_in['Variance_Y'] = var_Ys
	return dict_in


def dict_to_csv(dict_in, file_name):
	df = pd.DataFrame(data=dict_in)
	df.to_csv(file_name)



def cartesian_product(list1, n_times = 2):
	prod_list = list(itertools.product(list1, 
		repeat = n_times))
	np_out = np.asarray(prod_list)
	return np_out



def get_hamming_distances(compare_string, list_strings):
	return np.count_nonzero( 
		compare_string != list_strings, axis = 1 )



def direct_search_X_random(X_in, bit_vals = [-1, 1]):
	n_features = X_in.shape[1]
	X_arr = X_in.tolist()
	X_new = X_in[0].tolist()
	bit_flips = list(range(n_features))
	shuffle(bit_flips)
	for i in range(n_features):
		if X_new[bit_flips[i]] == bit_vals[0]:
			X_new[bit_flips[i]] = bit_vals[1]
		else:
			X_new[bit_flips[i]] = bit_vals[0]

		X_arr.append(X_new.copy())

	return np.asarray(X_arr[1:])


def direct_search_batch(model, X, Y_base_0, 
	batch_size = None, out_thresh = 0, n_features = 1000):
	if batch_size is None: # fix this in future
		Y = model.predict(X)
		Y_0 = Y > out_thresh

		if np.any((not Y_base_0) == Y_0):
			return np.where((not Y_base_0) == Y_0)[0][0]+1
		else:
			return n_features
	else:
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





def ave_min_hamming_distance(X_in, y_in, n_random = None):
	
	y_vals = np.unique(y_in)

	if len(y_vals) > 2:
		raise ValueError("Y must be binary (have only two values)")

	# if all y are same value then return log2 of the length of y's
	if len(y_vals) == 1:
		return np.log2(len(y_in))

	# get indices of first and second values of y
	y_0_ind = y_in == y_vals[0]
	y_1_ind = y_in == y_vals[1]

	X_0 = X_in[y_0_ind.flatten()]
	X_1 = X_in[y_1_ind.flatten()]

	if n_random is not None:
		p = n_random/len(y_in)
		X_0_inds = np.random.choice(
			X_0.shape[0], int(p*X_0.shape[0]))
		X_1_inds = np.random.choice(
			X_1.shape[0], int(p*X_1.shape[0]))
		X_0_sub = X_0[X_0_inds, :]
		X_1_sub = X_1[X_1_inds, :]


	min_vals = []

	for X_i in X_0_sub:
		dists = get_hamming_distances(X_i, X_1)
		min_vals.append(dists.min())

	for X_i in X_1_sub:
		dists = get_hamming_distances(X_i, X_0)
		min_vals.append(dists.min())

	return np.average(min_vals)


# calculates hamming distance for a given point X_in
def randomized_min_hamming_distance(model, X_in, y_in = None, 
	n_steps_max = 4, replace_vals = [0, 1], threshold = None):
	
	if y_in is None:
		y_in = model.predict(X_in.reshape(-1, len(X_in)))
	print(y_in)

	if threshold is not None:
		y_in = y_in > threshold

	X_bit = X_in

	if replace_vals is not None:
		X_bit[X_in == replace_vals[0]] = 0
		X_bit[X_in == replace_vals[1]] = 1

	step_i = 1
	no_change = True

	X_str = flatten_outputs(X_bit)

	while step_i <= n_steps_max and no_change:
		str_at_dist = hamming(X_str, step_i)
		dist_ints = [[int(i) for i in word] for word in str_at_dist]
		X_hamming = np.asarray(dist_ints)

		if replace_vals is not None:
			X_hamming[X_hamming == 0] = replace_vals[0]
			X_hamming[X_hamming == 1] = replace_vals[1]

		Y_new = model.predict(X_hamming)

		if threshold is not None:
			Y_new = Y_new > threshold

		if np.any((not y_in) == Y_new):
			no_change = False
		else:
			step_i+=1

	return step_i


# returns a list of lists at given hamming distance of input binary vector
def hamming_list(vec, dist, a = -1, b = 1):
	
	def append_correctly(item1, item2, item3):
		if isinstance(item3, list):
			if any(isinstance(el, list) for el in item3):
				for i in item3:
					outputlist.append(item1+item2+i)
			else:
				outputlist.append(item1+item2+item3)
		else:
			outputlist.append(item1+item2+[item3])


	if dist == 0:
		return vec
	else:
		outputlist = list()
		for item in range(len(vec)):
			if len(vec[item:]) > dist - 1:
				if vec[item] == a:
					rest = hamming_list(list(vec[item+1:]), dist - 1)
					if isinstance(rest, list):
						outputlist.append(vec[:item]+[b]+rest)
					else:
						outputlist.append(vec[:item]+[b]+[rest])
				else:
					rest = hamming_list(list(vec[item + 1:]), dist - 1)
					append_correctly(vec[:item],[a],rest)
	return outputlist




# returns a list of strings at given hamming distance of input binary string
def hamming(num, dist):
	#num must be a string
    if dist == 0:
        return num
    else:
        outputlist = list()
        for item in range(len(num)):
            if len(num[item:]) > dist - 1:
                if num[item] == "0":
                    restoflist = hamming(num[item + 1:], dist - 1)
                    if type(restoflist) is not str:
                        for rest in restoflist:
                            outputlist.append(num[:item] + "1" + str(rest))
                    else:
                        outputlist.append(num[:item] + "1" + str(restoflist))
                else:
                    restoflist = hamming(num[item + 1:], dist - 1)
                    if type(restoflist) is not str:
                        for rest in restoflist:
                            outputlist.append(num[:item] + "0" + str(rest))
                    else:
                        outputlist.append(num[:item] + "0" + str(restoflist))                
    return outputlist		



# hamming distance that is fast but not exact
# returns all numbers that are 2 steps away (steps may result in same output)
# assumes inputs are binary in format of {-1, 1}
def fast_hamming(start, dist,
	remove_duplicates= False):
	dim_n = len(start)

	# np array of possible changes as array to be multiplied element wise
	change_arr = np.ones((dim_n, dim_n), dtype=int)
	np.fill_diagonal(change_arr, -1)
	change_arr_0 = np.copy(change_arr)
	a_change_arr = np.copy(change_arr)

	# get changes at higher distances
	for i in range(dist-1):
		len_changes = a_change_arr.shape[0]
		a_change_arr = np.repeat(change_arr,
			np.repeat(dim_n, len_changes),
			axis = 0)

		b_change_arr = np.tile(change_arr_0,
			(len_changes, 1))

		change_arr = np.multiply(a_change_arr, b_change_arr)

	out_arr = np.multiply(start,change_arr)

	if remove_duplicates:
		out_arr = np.unique(out_arr, axis = 0)

	return out_arr



def greedy_search(model, X_in, Y_base = None, 
	bit_vals = [-1, 1], threshold = 0, max_search = None, 
	batch_pred_size = None,
	return_X = False):
	
	if Y_base is None:
		Y_base = model.predict(X_in)

	if max_search is None:
		max_search = X_in.shape[1]

	Y_new = Y_base[0][0]

	X_new = X_in
	dist = 0
	# print('starting search')
	while np.sign(Y_base-threshold) == np.sign(Y_new-threshold) and dist <= max_search:
		Y_new, X_new = greedy_bit_change(model, X_new, 
			bit_vals = bit_vals, threshold = threshold, 
			batch_size = batch_pred_size)
		dist += 1

	if return_X:
		return dist, X_new
	else:
		return dist



def greedy_bit_change(model, X_in, Y_base = None,
	bit_vals = [-1, 1], threshold = 0, batch_size = None):
	
	if Y_base is None:
		Y_base = model.predict(X_in)

	X_new = flip_all_bits(X_in, 
		a = bit_vals[0], b = bit_vals[1])
	# print(X_new)
	
	if batch_size is None:
		Y_new = model.predict(X_new)
	else:
		n_batches = ceil(X_new.shape[0] / batch_size)
		for i in range(n_batches):
			if i == 0:
				Y_new = model.predict(X_new[0:batch_size])
			elif i == (n_batches - 1):
				Y_new = np.append(Y_new, 
					model.predict(X_new[batch_size*i:]),
					axis = 0)
			else:
				Y_new = np.append(Y_new, 
					model.predict(X_new[batch_size*i:batch_size*(i+1)]),
					axis = 0)

	if Y_base[0] >= threshold:
		out_ind = np.argmin(Y_new, axis=0)
	else:
		out_ind = np.argmax(Y_new, axis=0)
	# print(Y_new[out_ind][0][0])
	return Y_new[out_ind][0][0], X_new[out_ind]


def flip_all_bits(vec, a = -1, b = 1, length_flips = None):
	if length_flips is None:
		length_flips = vec.shape[1]
	vec_expanded = np.tile(vec,(length_flips,1))

	for i in range(length_flips):
		if vec_expanded[i,i] == a:
			vec_expanded[i,i] = b
		else: 
			vec_expanded[i,i] = a

	return vec_expanded


def max_derivative_input(model, X_in, Y_base = None, 
	dX_step = 0.01):
	
	if Y_base is None:
		Y_base = model.predict(X_in)

	n_samples = X_in.shape[0]
	n_vars = X_in.shape[1]

	# expand X and Y by n_vars to calculate derivative in each variable
	X_expanded = np.repeat(X_in, 
		[n_vars]*n_samples, axis=0)
	Y_expanded = np.repeat(Y_base, 
		[n_vars]*n_samples, axis=0)

	# calculate derivatives by adding a bit to a single variable
	dX = np.identity(n_vars)*dX_step
	dX = np.tile(dX, (n_samples,1))
	X_plus_dX = X_expanded+dX

	# calculate y_dx and derivative
	y_dX = model.predict(X_plus_dX)
	derivative = (y_dX - Y_expanded) / dX_step
	derivative = abs(derivative)

	# take maximum of derivative along reshaped array where each 
	# element in new dimension is a vector of the derivatives in the gradient
	max_derivs = np.max(
		derivative.reshape(-1, n_vars), axis=1)

	return max_derivs


def direct_search_X(n_X = 100, bit_vals = [-1,1]):
	a = np.ones((n_X, n_X))
	a = np.tril(a)
	if bit_vals[0] != 0:
		a[a==1] = bit_vals[0]
		a[a==0] = bit_vals[1]
	else: 
		a[a==1] = max(bit_vals)+1
		a[a==0] = min(bit_vals)-1
		a[a==max(bit_vals)+1] = bit_vals[0]
		a[a==min(bit_vals)-1] = bit_vals[1]
	return a


def take_random_walk(x_start, n_steps = 20, bit_vals = [-1, 1]):

	x_start = list(x_start)
	x_arr = [x_start]

	flip_bit = np.random.choice(len(x_start), size = n_steps)
	for flip_i in flip_bit:
		x_new = x_arr[-1].copy()
		if x_new[flip_i] == bit_vals[0]:
			x_new[flip_i] = bit_vals[1]
		else:
			x_new[flip_i] = bit_vals[0]
		x_arr.append(x_new)

	del x_arr[0]
	return np.asarray(x_arr)


def random_walk_change(model, X_in, Y_base = None,
	n_combine = 20, max_steps = 500, threshold = 0,
	bit_vals = [-1, 1]):
	
	if len(X_in.shape) == 1:
		X_in = X_in.reshape(-1, X_in.shape[0])

	if Y_base is None:
		Y_base = model.predict(X_in)

	Y_start = Y_base >= threshold
	Y_end = Y_base < threshold

	step_num = 0
	found_change = False

	if len(X_in.shape) == 1:
		X = X_in.reshape(-1,X_in.shape)
	else:
		X = X_in

	while not found_change and step_num < max_steps:
		X = take_random_walk(X[-1], n_steps = n_combine,
			bit_vals = bit_vals)
		Y = model.predict(X)
		Y = Y > threshold
		Y_eval = Y == Y_end
		Y_eval = Y_eval.reshape(Y_eval.shape[0])
		found_change = np.any(Y_eval)
		if found_change:
			step_num += np.argmax(Y_eval)+1
		else:
			step_num += n_combine

	return step_num



