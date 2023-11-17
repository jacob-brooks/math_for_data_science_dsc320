# %% [markdown]
# Data Normalization Function

# %%
def normalize_vector(vector):
    min_val = min(vector)
    max_val = max(vector)
    
    normalized_vector = [(x - min_val) / (max_val - min_val) for x in vector]
    
    return normalized_vector

# %% [markdown]
# Data Standardization Function

# %%
def standardize_vector(vector):
    mean_val = sum(vector) / len(vector)
    std_dev = (sum((x - mean_val) ** 2 for x in vector) / len(vector)) ** 0.5
    
    standardized_vector = [(x - mean_val) / std_dev for x in vector]
    
    return standardized_vector

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
file_path = 'C:/Users/JacobBrooks/Downloads/calif_housing_data.csv'  # Replace with the actual path to your file
housing_data = pd.read_csv(file_path)

# Print the column names to inspect them
print("Column names:", housing_data.columns)

# (a) How many rows does this data set have?
num_rows = housing_data.shape[0]
print(f"(a) Number of rows: {num_rows}")

# (b) What is the target vector for your model?
target_vector = housing_data['median_house_value']

# (c) Create a new feature: total bedrooms divided by the number of households
housing_data['bedrooms_per_household'] = housing_data['total_bedrooms'] / housing_data['households']
print("(c) New feature 'bedrooms_per_household' created.")

# (d) Create a new data frame with three features: median age, median income, and the new feature
selected_features = ['housing_median_age', 'median_income', 'bedrooms_per_household']
new_dataframe = housing_data[selected_features]
print("(d) New data frame with selected features created.")

# (e) Data standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(new_dataframe)
standardized_dataframe = pd.DataFrame(standardized_data, columns=new_dataframe.columns)
print("(e) Data standardization applied to the features.")


