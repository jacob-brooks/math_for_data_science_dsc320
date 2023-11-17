# %%
import pandas as pd

# Load the data
df = pd.read_csv('C:/Users/JacobBrooks/Downloads/week10data/week10data/qb_stats.csv')

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['number'])

# Calculate mean
mean_values = numeric_columns.mean()
print("Mean values:\n", mean_values)

# Calculate standard deviation
std_dev_values = numeric_columns.std()
print("\nStandard Deviation values:\n", std_dev_values)

# %%
import matplotlib.pyplot as plt

# Create a histogram
plt.hist(df['yds'], bins=20, edgecolor='black')
plt.title('Histogram of Number of Yards')
plt.xlabel('Number of Yards')
plt.ylabel('Frequency')
plt.show()

# %%
# Create a boxplot
plt.boxplot(df['td'], vert=False)
plt.title('Boxplot of Number of Touchdowns')
plt.xlabel('Number of Touchdowns')
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('C:/Users/JacobBrooks/Downloads/week10data/week10data/survey_data.csv')

# (a) Probability a college student has brown hair
total_students = data.shape[0]
brown_hair_students = data[data['hair_color'] == 'Brown'].shape[0]
prob_brown_hair = brown_hair_students / total_students if total_students != 0 else 0

# (b) Probability a college student has blue eyes
blue_eyes_students = data[data['eye_color'] == 'Blue'].shape[0]
prob_blue_eyes = blue_eyes_students / total_students if total_students != 0 else 0

# (c) Probability a college student has blue eyes given brown hair
brown_hair_students = data[data['hair_color'] == 'Brown']
prob_blue_eyes_given_brown_hair = brown_hair_students[data['eye_color'] == 'Blue'].shape[0] / brown_hair_students.shape[0] if brown_hair_students.shape[0] != 0 else 0

# (d) Probability a college student has brown hair given blue eyes
blue_eyes_students = data[data['eye_color'] == 'Blue']
prob_brown_hair_given_blue_eyes = blue_eyes_students[data['hair_color'] == 'Brown'].shape[0] / blue_eyes_students.shape[0] if blue_eyes_students.shape[0] != 0 else 0

# (e) Check if brown hair and blue eyes are independent
independence_check = prob_blue_eyes_given_brown_hair == prob_blue_eyes and prob_brown_hair_given_blue_eyes == prob_brown_hair

# (f) Create a bar graph
hair_color_counts = data['hair_color'].value_counts()
eye_color_counts = data['eye_color'].value_counts()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

hair_color_counts.plot(kind='bar', ax=axes[0], color='brown', edgecolor='black')
axes[0].set_title('Hair Color Distribution')
axes[0].set_xlabel('Hair Color')
axes[0].set_ylabel('Count')

eye_color_counts.plot(kind='bar', ax=axes[1], color='blue', edgecolor='black')
axes[1].set_title('Eye Color Distribution')
axes[1].set_xlabel('Eye Color')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

# Display results
print(f"(a) Probability of brown hair: {prob_brown_hair:.2f}")
print(f"(b) Probability of blue eyes: {prob_blue_eyes:.2f}")
print(f"(c) Probability of blue eyes given brown hair: {prob_blue_eyes_given_brown_hair:.2f}")
print(f"(d) Probability of brown hair given blue eyes: {prob_brown_hair_given_blue_eyes:.2f}")
print(f"(e) Independence check: {independence_check}")


