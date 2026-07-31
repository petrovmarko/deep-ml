def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	N, M = len(matrix), len(matrix[0])
	out = []
	for i in range((N if mode == 'row' else M)):
		res = []
		for j in range((M if mode == 'row' else N)):
			res.append(matrix[i][j] if mode == 'row' else matrix[j][i])
		out.append(sum(res) / len(res))
	return out