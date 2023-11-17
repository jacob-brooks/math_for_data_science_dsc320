# %%
import pandas as pd

# Load the data
data = pd.read_csv('C:/Users/JacobBrooks/Downloads/week3data/week3data/us_pop_data.csv')

# Create a new column for years since 1790
data['Years Since 1790'] = data['year'] - 1790

# Create a new column for population in millions
data['Population (Millions)'] = data['us_pop'] / 1e6

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(data['Years Since 1790'], data['Population (Millions)'])
plt.xlabel('Years Since 1790')
plt.ylabel('Population (Millions)')
plt.title('US Population Growth Over Time')
plt.grid(True)
plt.show()

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create the linear regression model
model = LinearRegression()

# Fit the model
X = data[['Years Since 1790']]
y = data['Population (Millions)']
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate R-squared value
r2 = r2_score(y, y_pred)
print(f'R-squared value: {r2}')

# %%
data['Years Squared'] = data['Years Since 1790'] ** 2

# %%
# Create the linear regression model with squared feature
model_squared = LinearRegression()

# Fit the model
X_squared = data[['Years Squared']]
model_squared.fit(X_squared, y)

# Make predictions
y_pred_squared = model_squared.predict(X_squared)

# Calculate R-squared value
r2_squared = r2_score(y, y_pred_squared)
print(f'R-squared value with squared feature: {r2_squared}')

# %%
plt.figure(figsize=(10, 6))
plt.plot(data['Years Since 1790'], y, label='Actual Population (Millions)')
plt.plot(data['Years Since 1790'], y_pred, label='Linear Regression Model')
plt.plot(data['Years Since 1790'], y_pred_squared, label='Linear Regression Model with Squared Feature')
plt.xlabel('Years Since 1790')
plt.ylabel('Population (Millions)')
plt.title('US Population Growth and Regression Models')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Comparing the two models and their R-squared values:
# 
# The linear regression model with the squared feature tends to fit the data better because it allows for a more flexible curve, which can capture the non-linear growth pattern in the population data.
# 
# This is apparent in the R-squared values: The R-squared value for the model with the squared feature (e.g., population = a * (Years Since 1790)^2 + b) will typically be higher than the R-squared value for the linear model (e.g., population = a * Years Since 1790 + b) because it can better explain the variance in the data by allowing for curvature.


