import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	data = data.astype(np.float64)
	stand = (data - data.mean(axis=0)) / data.std(axis=0)
	norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
	return stand, norm