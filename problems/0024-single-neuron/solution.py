import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	import numpy as np
	features = np.array(features)
	labels = np.array(labels)
	weights = np.array(weights)
	bias = np.array(bias)
	def sigmoid(x):
  		return 1 / (1 + math.exp(-x))

	ans = np.apply_along_axis(lambda x : sigmoid(weights[0] * x[0] + weights[1] * x[1] + bias), 1, features)
	return ans, ((ans - labels) ** 2).mean()