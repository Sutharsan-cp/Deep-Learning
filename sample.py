import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
x = np.random.uniform(-5, 5, 100)
x = np.sort(x)  # for nicer plotting

# Activation functions
def sigmoid(z):    return 1 / (1 + np.exp(-z))
def tanh(z):       return np.tanh(z)
def relu(z):       return np.maximum(0, z)
def leaky_relu(z, alpha=0.1): return np.where(z > 0, z, alpha * z)
def elu(z, alpha=1.0): return np.where(z > 0, z, alpha * (np.exp(z) - 1))

# Compute outputs
y_sigmoid   = sigmoid(x)
y_tanh      = tanh(x)
y_relu      = relu(x)
y_leakyrelu = leaky_relu(x)
y_elu       = elu(x)

# Plot
plt.figure(figsize=(12, 7))
plt.plot(x, y_sigmoid,   label='Sigmoid', linewidth=2)
plt.plot(x, y_tanh,      label='Tanh', linewidth=2)
plt.plot(x, y_relu,      label='ReLU', linewidth=2)
plt.plot(x, y_leakyrelu, label='Leaky ReLU (α=0.1)', linewidth=2)
plt.plot(x, y_elu,       label='ELU (α=1)', linewidth=2)

plt.axhline(0, color='gray', linestyle='--', alpha=0.4)
plt.axvline(0, color='gray', linestyle='--', alpha=0.4)
plt.title("Comparison of Activation Functions (-5 to 5)", fontsize=14)
plt.xlabel("Input value")
plt.ylabel("Output")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()