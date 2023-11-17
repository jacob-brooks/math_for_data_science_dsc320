# %%
import numpy as np

# Define the matrix A
A = np.array([[4, 0, 0],
              [-1, -6, 0],
              [0, 1, -2]])

# Define the vector v
v = np.array([2, 3, 0])

# (a) Check if v is an eigenvector of A
result_a = np.allclose(A.dot(v), 2 * v)
print("(a) Is the vector [2, 3, 0] an eigenvector of A? ", result_a)

# (b) Check if [1, 0, 0] is an eigenvector of A
v_b = np.array([1, 0, 0])
result_b = np.allclose(A.dot(v_b), 1 * v_b)
print("(b) Is the vector [1, 0, 0] an eigenvector of A? ", result_b)

# (c) Find all eigenvalues and eigenvectors of A
eigenvalues, eigenvectors = np.linalg.eig(A)
print("(c) Eigenvalues of A:", eigenvalues)
print("(c) Eigenvectors of A:", eigenvectors)


# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv('C:/Users/JacobBrooks/Downloads/archive (2)/video_game_data.csv')

# Extract critic scores and user scores
critic_scores = data['Critic_Score']
user_scores = data['User_Score']

# (a) Make a scatterplot of user scores versus critic scores
plt.scatter(critic_scores, user_scores)
plt.title('Scatterplot of User Scores vs Critic Scores')
plt.xlabel('Critic Scores')
plt.ylabel('User Scores')
plt.show()

# (b) Perform PCA to find principal components
# Combine critic scores and user scores into a matrix
scores_matrix = np.column_stack((critic_scores, user_scores))

# Fit PCA
pca = PCA()
pca.fit(scores_matrix)

# Get principal components
components = pca.components_

# Plot the scatterplot with principal components
plt.scatter(critic_scores, user_scores, alpha=0.5)
for i in range(2):
    plt.arrow(pca.mean_[0], pca.mean_[1], components[i, 0], components[i, 1], color='red', linewidth=2)
plt.title('Scatterplot with Principal Components')
plt.xlabel('Critic Scores')
plt.ylabel('User Scores')
plt.show()



