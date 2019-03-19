import time
import numpy as np
from itertools import permutations

start_time = time.time()

vals = range(1, 10)
omega = np.sum(vals, axis=0) / 3

a = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 0]
    ]
)
solutions = []
for mat in permutations(vals):
    A = a.dot(np.array(mat).reshape((9, 1)))
    if np.all(np.where(A == omega, 1, 0)):
        solutions.append(mat)

print(solutions)
print(time.time() - start_time)
