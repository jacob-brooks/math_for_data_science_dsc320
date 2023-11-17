# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("C:/Users/JacobBrooks/Downloads/week2data/week2data/car_data.csv")

# Scatterplot
plt.scatter(data["weight"], data["hwy_mpg"])
plt.xlabel("Weight (lbs)")
plt.ylabel("Highway MPG")
plt.title("Scatterplot of Highway MPG vs. Weight")
plt.show()

# %% [markdown]
# 2. Trend of Highway MPG with Weight:
# 
# Based on the scatterplot (not provided here, but you can create it using the code in question 1), it appears that there is a general trend of decreasing highway miles per gallon (MPG) as the weight of the vehicle increases. This suggests a negative correlation between weight and highway MPG.
# 
# 3. Expectation of Slope:
# 
# Given the trend observed in the scatterplot, you would expect the slope of a linear model predicting highway miles per gallon from weight to be negative. In other words, as the weight of the vehicle increases, you would expect the highway MPG to decrease.
# 
# 4. Interpretation of Slope = -0.05:
# 
# A slope of -0.05 would mean that for every one-unit increase in weight (in pounds), you would expect a decrease of 0.05 units in highway miles per gallon. In practical terms, this suggests that as the weight of the vehicle increases by 100 pounds, you would expect a decrease of 5 units in highway miles per gallon. This indicates that heavier vehicles tend to have lower fuel efficiency on the highway.

# %%
# Scatterplot with a manually adjusted line
plt.scatter(data["weight"], data["hwy_mpg"])
plt.xlabel("Weight (lbs)")
plt.ylabel("Highway MPG")
plt.title("Scatterplot of Highway MPG vs. Weight")

# Manually adjusted line
slope = -0.05  # Adjust as needed
y_intercept = 40  # Adjust as needed
plt.plot(data["weight"], slope * data["weight"] + y_intercept, color='red')

plt.show()

# %%
pip install scikit-learn

# %%
from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Fit the model to your data
X = data[["weight"]]
y = data["hwy_mpg"]
model.fit(X, y)

# Get the slope and y-intercept of the best-fit line
best_fit_slope = model.coef_[0]
best_fit_intercept = model.intercept_

print(f"Best-fit line: y = {best_fit_slope:.2f}x + {best_fit_intercept:.2f}")

# %%
from sklearn.metrics import mean_squared_error
import numpy as np

# Predictions from the manually adjusted line
manual_predictions = slope * data["weight"] + y_intercept

# Predictions from the best-fit line
best_fit_predictions = model.predict(X)

# Calculate RMSE for both
rmse_manual = np.sqrt(mean_squared_error(y, manual_predictions))
rmse_best_fit = np.sqrt(mean_squared_error(y, best_fit_predictions))

print(f"RMSE for manually adjusted line: {rmse_manual:.2f}")
print(f"RMSE for best-fit line: {rmse_best_fit:.2f}")

# %%
weight_to_predict = 3200
predicted_mpg = model.predict(np.array([[weight_to_predict]]))[0]

print(f"Predicted Highway MPG for a car weighing 3200 pounds: {predicted_mpg:.2f}")


