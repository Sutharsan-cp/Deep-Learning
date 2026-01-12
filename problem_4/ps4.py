import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 400)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

plt.figure(figsize=(9,5))
plt.plot(x, sigmoid, label='Sigmoid', linewidth=2)
plt.plot(x, tanh, label='Tanh', linewidth=2)
plt.axhline(0, color='gray', linestyle='--', alpha=0.4)
plt.legend()
plt.title("Sigmoid vs Tanh [-10, 10]")
plt.grid(True, alpha=0.3)
plt.show()