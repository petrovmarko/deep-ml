import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	# Your code here
	padded_matrix = np.pad(input_matrix, pad_width = padding)
	lista = []
	for i in range(0,padded_matrix.shape[0]+1-kernel.shape[0], stride):
		l = []
		for j in range(0,padded_matrix.shape[1]+1-kernel.shape[1], stride):
			l.append((padded_matrix[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel).sum())
		lista.append(l)
	return lista
	
