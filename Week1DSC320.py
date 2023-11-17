# %%
import numpy as np

def rmse(actual, predicted):
    """
    Calculate the Root Mean Square Error (RMSE) between actual and predicted values.

    Parameters:
    actual (numpy array): Array of actual values.
    predicted (numpy array): Array of predicted values.

    Returns:
    float: RMSE value.
    """
    # Check if the input arrays have the same length
    if len(actual) != len(predicted):
        raise ValueError("Input arrays must have the same length.")
    
    # Calculate the squared differences
    squared_diff = (predicted - actual) ** 2
    
    # Calculate the mean of squared differences and take the square root
    rmse_value = np.sqrt(np.mean(squared_diff))
    
    return rmse_value

# %%
import pandas as pd

# Load the housing data
housing_data = pd.read_csv('C:/Users/JacobBrooks/Downloads/week1data/week1data/housing_data.csv')

# Extract the actual and predicted sale prices
actual_prices = housing_data['sale_price']
predicted_prices = housing_data['sale_price_pred']

# Calculate RMSE
rmse_result = rmse(actual_prices, predicted_prices)

# Print the RMSE result
print("RMSE:", rmse_result)

# %%
def mae(actual, predicted):
    """
    Calculate the Mean Absolute Error (MAE) between actual and predicted values.

    Parameters:
    actual (numpy array): Array of actual values.
    predicted (numpy array): Array of predicted values.

    Returns:
    float: MAE value.
    """
    # Check if the input arrays have the same length
    if len(actual) != len(predicted):
        raise ValueError("Input arrays must have the same length.")
    
    # Calculate the absolute differences
    absolute_diff = np.abs(predicted - actual)
    
    # Calculate the mean of absolute differences
    mae_value = np.mean(absolute_diff)
    
    return mae_value

# %%
# Calculate MAE
mae_result = mae(actual_prices, predicted_prices)

# Print the MAE result
print("MAE:", mae_result)

# %%
def accuracy(actual, predicted):
    """
    Calculate the accuracy between actual and predicted binary values.

    Parameters:
    actual (numpy array): Array of actual binary values.
    predicted (numpy array): Array of predicted binary values.

    Returns:
    float: Accuracy as a percentage.
    """
    # Check if the input arrays have the same length
    if len(actual) != len(predicted):
        raise ValueError("Input arrays must have the same length.")
    
    # Calculate the number of correct predictions
    correct_predictions = np.sum(actual == predicted)
    
    # Calculate accuracy as a percentage
    accuracy_value = (correct_predictions / len(actual)) * 100.0
    
    return accuracy_value

# %%
# Load the mushroom data
mushroom_data = pd.read_csv('C:/Users/JacobBrooks/Downloads/week1data/week1data/mushroom_data.csv')

# Extract the actual and predicted values
actual_values = mushroom_data['actual']
predicted_values = mushroom_data['predicted']

# Calculate accuracy
accuracy_result = accuracy(actual_values, predicted_values)

# Print the accuracy result
print("Accuracy:", accuracy_result, "%")

# %%
pip install matplotlib

# %%
import matplotlib.pyplot as plt

# Define the error function
def error_function(p):
    return 0.005*p**6 - 0.27*p**5 + 5.998*p**4 - 69.919*p**3 + 449.17*p**2 - 1499.7*p + 2028

# Generate a range of values for p
p_values = np.linspace(-10, 10, 1000)

# Calculate the corresponding error values
error_values = error_function(p_values)

# Plot the error function
plt.figure(figsize=(8, 6))
plt.plot(p_values, error_values)
plt.title("Error Function")
plt.xlabel("Parameter (p)")
plt.ylabel("Error")
plt.grid(True)
plt.show()


