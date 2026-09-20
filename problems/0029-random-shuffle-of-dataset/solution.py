import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
	X = np.array(X)
	y = np.array(y)
	G = np.random.RandomState(seed)
	ids = G.permutation(len(X))
	return X[ids], y[ids]