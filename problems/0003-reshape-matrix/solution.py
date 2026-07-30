import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	lista = [a[i][j] for i in range(len(a)) for j in range(len(a[i]))]
	if len(lista) != new_shape[0] * new_shape[1]:
		return []
	
	out = [[0] * new_shape[1] for _ in range(new_shape[0])]
	for i in range(len(lista)):
		out[i // new_shape[1]][i % new_shape[1]] = lista[i]
	return out