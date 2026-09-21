import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	y = X @ w
	L = ((y - y_true) ** 2).mean() + (w ** 2).sum() * alpha
	return L
