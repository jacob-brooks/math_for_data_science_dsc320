# %%
pip install numpy

# %%
import numpy as np

# Define matrices A, B, and C
A = np.array([[1, -2, 3, 7],
              [2, 1, 1, 4],
              [-3, 2, -2, 10]])

B = np.array([[0, 1, -3, -2],
              [10, -1, 2, -3],
              [5, 1, -1, 4]])

C = np.array([[4, 0, -2, 3],
              [3, 6, 9, 7],
              [2, 2, 5, 1],
              [9, 4, 6, -2]])

# 2. Dimensions of matrices A, B, and C
dims_A = A.shape
dims_B = B.shape
dims_C = C.shape
print(f"Dimensions of A: {dims_A}")
print(f"Dimensions of B: {dims_B}")
print(f"Dimensions of C: {dims_C}")

# 3. Matrix Operations
# (a) A + B
if A.shape == B.shape:
    result_a = A + B
else:
    result_a = "Not defined, shapes do not match."

# (b) AB
if A.shape[1] == B.shape[0]:
    result_b = np.dot(A, B)
else:
    result_b = "Not defined, inner dimensions do not match."

# (c) AC
if A.shape[1] == C.shape[0]:
    result_c = np.dot(A, C)
else:
    result_c = "Not defined, inner dimensions do not match."

# (d) C^T
result_d = C.T

# 4. Property of matrix transpositions: (A + B)^T = A^T + B^T
result_4 = (A + B).T == A.T + B.T

# 5. Property of matrix transpositions: (AC)^T = C^T * A^T
result_5 = (np.dot(A, C)).T == np.dot(C.T, A.T)

# 6. Inverse of C
try:
    C_inv = np.linalg.inv(C)
except np.linalg.LinAlgError:
    C_inv = "Matrix is not invertible."

# Print results
print("\n(a) A + B:")
print(result_a)
print("\n(b) AB:")
print(result_b)
print("\n(c) AC:")
print(result_c)
print("\n(d) C^T:")
print(result_d)
print("\n4. Property of matrix transpositions: (A + B)^T = A^T + B^T:")
print(result_4)
print("\n5. Property of matrix transpositions: (AC)^T = C^T * A^T:")
print(result_5)
print("\n6. Inverse of C:")
print(C_inv)

# %%
pip install opencv-python

# %%
#import the OpenCV library to read images
import cv2
#import matplotlib to display the image
import matplotlib.pyplot as plt

# Specify the file path to your image
image_path = 'C:/Users/JacobBrooks/Downloads/week7data/week7data/week7_image.jpg'

# Import the image; by default, OpenCV imports images in the BGR format
image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Check if the image loaded successfully
if image_bgr is not None:
    # Convert the image to the RGB format
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()
else:
    print("Failed to load the image. Check the file path and image format.")

# %%
image_resolution = image_rgb.shape
height, width, channels = image_resolution
print("Image resolution (width x height):", width, "x", height)

# %% [markdown]
# When you check the array dimensions using image_rgb.shape, you'll see three numbers. These numbers represent the following:
# 
# The first number is the height of the image in pixels.
# The second number is the width of the image in pixels.
# The third number is the number of color channels in the image.
# 
# In the code provided, the image is imported with the cv2.IMREAD_COLOR flag, which means it's read as a 3-channel color image. The third number in the shape corresponds to the number of color channels, which is typically 3 for a color image (Red, Green, and Blue channels). If it were a grayscale image, you would see 1 instead of 3, indicating a single grayscale channel.


