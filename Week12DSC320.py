# %%
# Given probabilities
P_A = 0.5
P_B = 0.3
P_C = 0.2

P_D_given_A = 0.03
P_notD_given_A = 0.97

P_D_given_B = 0.02
P_notD_given_B = 0.98

P_D_given_C = 0.04
P_notD_given_C = 0.96

# Total probabilities
P_D = P_D_given_A * P_A + P_D_given_B * P_B + P_D_given_C * P_C
P_notD = 1 - P_D  # Probability of not being defective

# (a) Probability that a randomly chosen graphics card is defective and manufactured using Process A
P_A_given_D = (P_D_given_A * P_A) / P_D

# (b) Probability that a randomly chosen graphics card is not defective and manufactured using Process C
P_C_given_notD = (P_notD_given_C * P_C) / P_notD

# Output the results
print(f"(a) Probability that a randomly chosen graphics card is defective and manufactured using Process A: {P_A_given_D:.4f}")
print(f"(b) Probability that a randomly chosen graphics card is not defective and manufactured using Process C: {P_C_given_notD:.4f}")


# %%
import math

def entropy(probabilities):
    # Ensure that the probabilities sum up to 1
    if not math.isclose(sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum up to 1.")

    # Calculate entropy using the formula
    entropy_value = -sum(p * math.log2(p) if p != 0 else 0 for p in probabilities)
    
    return entropy_value

# Example usage:
probabilities = [0.2, 0.3, 0.5]
result = entropy(probabilities)
print(f"Entropy: {result}")


# %%
import math

def entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities)

# Given probabilities for X
probabilities_X = [0.2, 0.2, 0.2, 0.2, 0.2]

# Given probabilities for Y
probabilities_Y = [0.1, 0.4, 0.1, 0.3, 0.1]

# Calculate entropies
entropy_X = entropy(probabilities_X)
entropy_Y = entropy(probabilities_Y)

# Print the results
print("Entropy of X:", entropy_X)
print("Entropy of Y:", entropy_Y)



