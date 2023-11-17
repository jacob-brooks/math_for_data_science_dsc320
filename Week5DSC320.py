# %%
def right_riemann_sum(f, a, b, n):
    """
    Calculate the right Riemann sum of a function f on the interval [a, b] with n rectangles.

    Parameters:
    - f: The mathematical function.
    - a: The left endpoint of the interval.
    - b: The right endpoint of the interval.
    - n: The number of rectangles.

    Returns:
    The right Riemann sum.
    """
    delta_x = (b - a) / n
    sum_result = 0

    for k in range(1, n + 1):
        x_k = a + k * delta_x
        sum_result += f(x_k) * delta_x

    return sum_result

# Example usage:
# Define the function you want to integrate
def my_function(x):
    return x**2

# Set the interval [a, b] and the number of rectangles (n)
a = 0
b = 2
n = 4

# Calculate the right Riemann sum
result = right_riemann_sum(my_function, a, b, n)

# Print the result
print("Right Riemann sum:", result)

# %%
def my_function(x):
    return 4 - x**2

a = 0
b = 2
n = 4

result = right_riemann_sum(my_function, a, b, n)
print("Right Riemann sum R4:", result)


# %% [markdown]
# ### (c) Finding the area under the curve using definite integral:
# 
# To find the area under the curve, we can calculate the definite integral of \( f(x) = 4 - x^2 \) over the interval \([0, 2]\) using the limit of the right Riemann sum as \( n \) approaches infinity:
# 
# \[ \text{Area Under the Curve} = \lim_{n \to \infty} R_n \]
# 
# In practice, we can use a large enough value of \( n \) to get a good approximation of the area.
# 
# The area under the curve, denoted as:
# 
# \[ \int_{0}^{2} (4 - x^2) \,dx \]
# 
# can be calculated using numerical integration methods or Python libraries such as `scipy.integrate.quad`. For the specific case of \( n = 4 \), we already calculated the right Riemann sum \( R_4 \):
# 
# \[ R_4 = \text{{value calculated in part (b)}} \]
# 
# This value provides an approximation of the area under the curve for the given function and interval.
# 


