# %%
import numpy as np

# Define the coefficient matrix A and the right-hand side vector b
A = np.array([[27, -10, 4, -29],
              [-16, 5, -2, 18],
              [-17, 4, -2, 20],
              [-7, 2, -1, 8]])

b = np.array([1, -1, 0, 1])

# Solve for x using numpy.linalg.solve
x = np.linalg.solve(A, b)

# Print the solution
print("Solution for x:")
print(x)

# %%
# Define the matrix A
A = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2],
              [np.sqrt(2)/2, np.sqrt(2)/2]])

# Define the vector x
x = np.array([-2, 2])

# Calculate T(x) as Ax
Tx = np.dot(A, x)

# Print the result
print("T(x) = ", Tx)

# %%
import matplotlib.pyplot as plt

# Define the matrix A
A = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2],
              [np.sqrt(2)/2, np.sqrt(2)/2]])

# Create a grid of points in the input space R^2
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)
input_vectors = np.vstack((X.flatten(), Y.flatten()))

# Apply the linear transformation to all input vectors
output_vectors = np.dot(A, input_vectors)

# Reshape the output vectors back to a grid
X_out = output_vectors[0, :].reshape(X.shape)
Y_out = output_vectors[1, :].reshape(Y.shape)

# Plot the input and output vectors
plt.quiver(X, Y, X_out, Y_out, angles='xy', scale_units='xy', scale=1, color=['r', 'b'])

# Set axis limits and labels
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y')

plt.title('Linear Transformation T(x) = Ax')
plt.grid(True)
plt.show()

# %% [markdown]
# (c) Geometrically, the linear transformation T takes vectors in R^2 and rotates them counterclockwise by 45 degrees while preserving their length. In other words, it's a rotation transformation with a 45-degree angle. It does not change the length of vectors; it only changes their direction.


