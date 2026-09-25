
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	rmse_res = ((y_true - y_pred) ** 2).mean() ** 0.5
	return round(rmse_res,3)
