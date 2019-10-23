import numpy as np
import pickle
import time


# returns a list of lists at given hamming distance of input binary vector
def hamming_list(vec, dist, a = -1, b = 1):
	
	# list of lists need to be appended correctly
	def append_correctly(item1, item2, item3):
		if isinstance(item3, list):
			if any(isinstance(el, list) for el in item3):
				for i in item3:
					outputlist.append(item1+item2+i)
			else:
				outputlist.append(item1+item2+item3)
		else:
			outputlist.append(item1+item2+[item3])

	# recursively create hamming list
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


X_val_arr = []
vec = [1]*100			# input number of features here for hamming list to be created

for i in range(3):
	X_val_arr.append(np.asarray(hamming_list(vec, i+1)))
	print(i)

	with open('../pickles/hamming_list_100_3.pickle', 'wb') as handle:
	    pickle.dump(X_val_arr, handle)
