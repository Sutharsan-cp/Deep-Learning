import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input vector
X = np.array([1, 0, 1])

# Hidden layer weights (2 neurons × 3 inputs)
W_hidden = np.array([
    [0.5, -0.2, 0.3],   # weights for H1
    [0.4,  0.1, -0.5]   # weights for H2
])

b_hidden = np.array([0, 0])  # bias = 0 for both hidden neurons

# Output layer weights (1 output × 2 hidden)
W_output = np.array([0.7, 0.2])
b_output = 0

# -----------------------------
# Step 1: Hidden layer (pre-activation)
z_hidden = np.dot(X, W_hidden.T) + b_hidden
# z_hidden = [z1, z2]

print("Hidden layer pre-activation (z):", z_hidden)

# Step 2: Apply ReLU activation
a_hidden = relu(z_hidden)
print("Hidden layer activation (a):   ", a_hidden)

# -----------------------------
# Step 3: Output layer (pre-activation)
z_output = np.dot(a_hidden, W_output) + b_output
print("Output pre-activation (z):     ", z_output)

# Step 4: Apply Sigmoid activation
output = sigmoid(z_output)
print("\nFinal output (probability):   {:.6f}".format(output))

# Rounded version (common for binary classification display)
print("Rounded prediction:            ", round(output))