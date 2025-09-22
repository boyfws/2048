from numba import njit 
import numba
import numpy as np 

@njit(
        numba.int32[:](numba.int32[:])
)
def move2right_line(line: np.ndarray)-> np.ndarray:
    non_zero = line[line != 0]
    new_line = np.concatenate(
        (
            np.zeros(len(line) - len(non_zero), dtype=line.dtype), 
            non_zero
        )
    )

    for i in range(len(new_line)-1, 0, -1):
        if new_line[i] == new_line[i-1] and new_line[i] != 0:
            new_line[i] *= 2
            new_line[i-1] = 0

    non_zero = new_line[new_line != 0]
    new_line = np.concatenate(
        (np.zeros(len(new_line) - len(non_zero), dtype=line.dtype), non_zero)
    )

    return new_line


@njit(numba.int32[:, :](numba.int32[:, :]))
def move2right(
    matrix: np.ndarray
) -> np.ndarray:
    new_matrix = np.zeros_like(matrix, dtype=np.int32)
    for i in range(matrix.shape[0]):
        new_matrix[i] = move2right_line(matrix[i])

    return new_matrix
