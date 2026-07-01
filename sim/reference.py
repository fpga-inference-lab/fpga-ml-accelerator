import numpy as np


def to_q44(value):
   return int(round(value * 16))


def from_q44(value):
   return value / 16


B = np.array([
   [1, 1, 1, 1],
   [1, 1, 1, 1],
   [1, 1, 1, 1],
   [1, 1, 1, 1]
])


A = np.array([
   [2, 3, 1, 0],
   [1, 4, 2, 3],
   [0, 2, 5, 1],
   [3, 1, 0, 4]
])


result = A @ B


print("Input matrix A:")
print(A)
print("\nWeight matrix B:")
print(B)
print("\nResult (A x B):")
print(result)

