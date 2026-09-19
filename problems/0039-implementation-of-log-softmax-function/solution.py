import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	scores = np.array(scores)
	softmax = np.exp(scores) / np.exp(scores).sum()
	return np.log(softmax)