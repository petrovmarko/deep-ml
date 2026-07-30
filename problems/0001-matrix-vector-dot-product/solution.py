def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	N, M = len(a), len(a[0])
	# (N, M) @ (M, 1) -> (N, 1)
	if M != len(b):
		return -1
	out = []
	for i in range(N):
		res = 0
		for j in range(M):
			res += a[i][j] * b[j]
		out.append(res)
	return out