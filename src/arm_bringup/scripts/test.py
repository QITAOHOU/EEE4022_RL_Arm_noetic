import numpy as np

test  = np.array([1.0, 2.0, 3.0])

test2  = np.array([[1.0, 2.0, 3.0]])

print(np.shape(test))

print(np.shape(test2.reshape(3,)))