import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	N = len(X)
	ids = np.arange(0, N)
	#np.random.shuffle(ids)
	ans = []
	for i in range(0, N, batch_size):
		if y is not None:
			batch = [list(X[ids[i:i+batch_size]]), list(y[ids[i:i+batch_size]])]
		else:
			batch = [list(X[ids[i:i+batch_size]])]
		ans.append(batch)

	return ans