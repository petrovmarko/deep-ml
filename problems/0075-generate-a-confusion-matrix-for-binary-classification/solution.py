
from collections import Counter
import numpy as np
def confusion_matrix(data):
	# Implement the function here
	data = np.array(data)
	return [[np.all(data == [1,1], axis=1).sum(), np.all(data == [1,0], axis=1).sum()], [np.all(data == [0,1], axis=1).sum(), np.all(data == [0,0], axis=1).sum()]]
