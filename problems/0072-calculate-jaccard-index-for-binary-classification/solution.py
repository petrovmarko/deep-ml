
import numpy as np

def jaccard_index(y_true, y_pred):
	# Write your code here
	result = (y_true & y_pred).sum() / (y_true | y_pred).sum()
	return round(result, 3)
