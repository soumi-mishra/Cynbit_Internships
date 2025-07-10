import numpy as np

# Creating a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Array operations
print("Array + 2:\n", arr + 2)
print("Array squared:\n", arr ** 2)

# Slicing
print("First row:", arr[0])
print("Element at (1,2):", arr[1, 2])

# Reshaping
reshaped = arr.reshape((3, 2))
print("Reshaped (3,2):\n", reshaped)

# Broadcasting
ones = np.ones((2, 1))
print("Broadcasted sum:\n", arr + ones)

# Using attributes
print("Shape:", arr.shape)
print("Data type:", arr.dtype)