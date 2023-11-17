# %%
def f(x):
    return 2*x**3 - 4*x + 1

x_values = [2.9, 2.99, 2.999, 3.001, 3.01, 3.1]  # Values closer and closer to 3

limit_values = [f(x) for x in x_values]

print(limit_values)

# %%
import math

def g(x):
    return math.exp(x - 1) / x

x_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]  # Values closer and closer to 0

limit_values = [g(x) for x in x_values]

print(limit_values)

# %%
def average_rate_of_change(func, a, b):
    return (func(b) - func(a)) / (b - a)

# Example usage with f(x) = 3x^2, a = 1, and b = 4
result = average_rate_of_change(lambda x: 3*x**2, 1, 4)
print(result)

# %% [markdown]
# Average Rate of Change to Instantaneous Rate of Change:
# (a) To find the average speed between 5 and 6 seconds for f(t)=4.9t2, you can use the average_rate_of_change function.
# (b) Similarly, for the average speed between 5 and 5.5 seconds and (c) between 5 and 5.1 seconds.
# (d) To find the instantaneous speed at t=5 seconds, you can calculate the derivative of f(t) at t=5 using f′(t)=2⋅4.9t. Plug in t=5 to find the answer.
# (e) The derivative of f(t) is f′(t)=2⋅4.9t.
# (f) Evaluate the derivative at t=5 to find the instantaneous speed at t=5. Compare this value to the average speeds calculated in parts (a) - (c).
# Calculating and Interpreting Partial Derivatives:
# (a) To predict the selling price of a 5-year old car with a condition rating of 8, plug C=8 and Y=5 into the model equation P=16,000+2,400C−1,800Y and calculate the value of P:
# P=16,000+2,400⋅8−1,800⋅5
# Calculate this to find the predicted selling price.
# (b) To find ∂P/∂C, which represents the rate of change of the predicted selling price with respect to the condition rating (C), you need to take the partial derivative of the model equation P=16,000+2,400C−1,800Y with respect to C:
# ∂C∂P=2,400
# Interpretation: The value ∂P/∂C is 2,400, which means that for every one-unit increase in the condition rating (C) of the car, the predicted selling price (P) is expected to increase by $2,400, assuming all other factors remain constant.
# (c) To find ∂P/∂Y, which represents the rate of change of the predicted selling price with respect to the age of the car (Y), you need to take the partial derivative of the model equation P=16,000+2,400C−1,800Y with respect to Y:
# ∂Y∂P=−1,800
# Interpretation: The value ∂P/∂Y is -1,800, which means that for every one-year increase in the age of the car (Y), the predicted selling price (P) is expected to decrease by $1,800, assuming all other factors remain constant. This indicates that older cars tend to have lower predicted selling prices.


