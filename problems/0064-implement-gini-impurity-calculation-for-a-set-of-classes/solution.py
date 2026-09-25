
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	_, p = np.unique(y, return_counts=True)
	p = p.astype(np.float32)
	p /= len(y)
	return 1 - (p ** 2).sum()