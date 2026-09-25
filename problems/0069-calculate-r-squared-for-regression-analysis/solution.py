
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	down = (y_true - y_true.mean()) ** 2
	up = (y_pred - y_true) ** 2
	return 1 - up.sum() / down.sum()
