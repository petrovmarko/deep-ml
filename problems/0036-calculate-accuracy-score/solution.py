import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	acc = 0
	for x,y in zip(y_true, y_pred):
		acc += x == y 
	return acc / len(y_true)