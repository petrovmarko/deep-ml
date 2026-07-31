def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	N = len(vectors)
	M = len(vectors[0])
	meanV = [sum(x) / len(x) for x in vectors]
	out = [[0] * N for _ in range(N)]
	for i in range(N):
		for j in range(N):
			cov = 0
			for k in range(M):
				cov += (vectors[i][k] - meanV[i]) * (vectors[j][k] - meanV[j])
			cov /= M - 1
			out[i][j] = cov
	return out