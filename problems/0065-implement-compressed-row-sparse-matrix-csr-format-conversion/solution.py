import torch

def compressed_row_sparse_matrix(dense_matrix) -> tuple:
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation
    using PyTorch's built-in sparse CSR tensor support.

    :param dense_matrix: 2D list representing a dense matrix
    :return: A tuple containing (values tensor, column indices tensor, row pointer tensor)
    """
    lista = list(dense_matrix.view(-1))
    v, c, r = [], [], []
    for i, x in enumerate(lista):
        if x == 0:
            continue
        v.append(x)
        r.append(i/dense_matrix.shape[0])
        c.append(i % dense_matrix.shape[0])
    print("Values array:", v)
    print("Column indices array:", c)
    print("Row pointer array:", r)
