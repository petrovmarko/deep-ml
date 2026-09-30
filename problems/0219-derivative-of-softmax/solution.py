import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	N = len(x)
	mat = [[0] * N for _ in range(N)]
	exp = np.exp(np.array(x))
	q = exp / exp.sum()

	for i in range(N):
		for j in range(N):
			mat[i][j] = q[i] * ((i == j) -q[j])
	return mat