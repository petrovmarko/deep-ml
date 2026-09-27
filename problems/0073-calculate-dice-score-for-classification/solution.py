
import numpy as np

def dice_score(y_true, y_pred):
	# Write your code here
	if y_pred.sum() == 0 or y_true.sum() == 0:
		return 0
	precision = (y_pred & y_true).sum() / y_pred.sum()
	recall = (y_pred & y_true).sum() / y_true.sum()
	if (precision + recall == 0):
		return 0
	res = 2 * precision * recall / (recall + precision)
	return round(res, 3)
