# %%
from scipy.stats import binom

# (a) Ensemble model accuracy with 15 individual models each having 63% accuracy
individual_model_accuracy = 0.63
num_individual_models = 15

ensemble_model_accuracy = binom.pmf(range(num_individual_models + 1), num_individual_models, individual_model_accuracy).sum()
print(f"(a) Ensemble model accuracy: {ensemble_model_accuracy * 100:.2f}%")

# (b) Accuracy of individual models needed for ensemble model to have 95% accuracy
target_ensemble_accuracy = 0.95

# Binary search to find the accuracy of individual models
low, high = 0, 1
while high - low > 1e-6:
    mid = (low + high) / 2
    if binom.pmf(range(num_individual_models + 1), num_individual_models, mid).sum() >= target_ensemble_accuracy:
        high = mid
    else:
        low = mid

individual_model_accuracy_for_95 = int(high * 100)
print(f"(b) Individual model accuracy needed: {individual_model_accuracy_for_95}%")

# (c) Number of individual models needed for ensemble model to have 95% accuracy
target_individual_accuracy = 0.95

# Binary search to find the number of individual models
low, high = 1, 100
while high - low > 1:
    mid = (low + high) // 2
    if binom.pmf(range(mid + 1), mid, individual_model_accuracy).sum() >= target_individual_accuracy:
        high = mid
    else:
        low = mid + 1

num_individual_models_for_95 = low if low % 2 == 1 else low + 1  # Ensure odd number for clear majority
print(f"(c) Number of individual models needed: {num_individual_models_for_95}")


# %% [markdown]
# P(X=1)=0.1
# P(X=2)=0.2
# P(X=3)=0.3
# P(X=4)=0.2
# P(X=5)=0.2

# %%
import numpy as np

def generate_random_values():
    values = [1, 2, 3, 4, 5]
    probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]

    random_values = np.random.choice(values, size=50, p=probabilities)
    return np.mean(random_values)


# %%
def run_simulation():
    num_simulations = 1000
    means = [generate_random_values() for _ in range(num_simulations)]
    return means


# %%
import matplotlib.pyplot as plt

means = run_simulation()

plt.hist(means, bins=30, edgecolor='black')
plt.title('Histogram of Means')
plt.xlabel('Mean Value')
plt.ylabel('Frequency')
plt.show()


# %%
mean_of_means = np.mean(means)
std_dev_of_means = np.std(means)

print(f"Mean of means: {mean_of_means}")
print(f"Standard deviation of means: {std_dev_of_means}")



