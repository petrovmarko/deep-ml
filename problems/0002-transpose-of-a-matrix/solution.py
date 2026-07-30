def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    N, M = len(a), len(a[0])
    out = [[0] * N for _ in range(M)]
    for i in range(N):
        for j in range(M):
            out[j][i] = a[i][j]

    return out