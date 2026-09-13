def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	import numpy as np
	B = np.array(B)
	C = np.array(C)
	return np.linalg.inv(C) @ B