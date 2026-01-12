import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Generate non-linear toy data
np.random.seed(42)
X = np.random.uniform(-5, 5, 400).reshape(-1, 1)
y = np.sin(1.2 * X) + 0.1 * X**2 + np.random.normal(0, 0.15, X.shape)

# 2. Simple 1-hidden-layer network (from scratch)
class TinyNet:
    def __init__(self, activation):
        self.activation = activation
        self.act_deriv = self._get_derivative()
        np.random.seed(42)
        self.W1 = np.random.randn(1, 10) * 0.3
        self.b1 = np.zeros((1, 10))
        self.W2 = np.random.randn(10, 1) * 0.3
        self.b2 = np.zeros((1, 1))
    
    def _get_derivative(self):
        if self.activation == 'sigmoid': return lambda x: x * (1 - x)
        if self.activation == 'tanh':    return lambda x: 1 - x**2
        if self.activation == 'relu':    return lambda x: (x > 0).astype(float)
        if self.activation == 'leaky_relu': return lambda x: np.where(x > 0, 1, 0.01)
        return lambda x: x  # fallback
    
    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2
    
    def train(self, X, y, lr=0.02, epochs=1200):
        losses = []
        for epoch in range(epochs):
            # Forward
            pred = self.forward(X)
            error = pred - y
            
            # Backward
            dW2 = self.a1.T @ error
            db2 = np.sum(error, axis=0, keepdims=True)
            d_hidden = error @ self.W2.T * self.act_deriv(self.a1)
            
            dW1 = X.T @ d_hidden
            db1 = np.sum(d_hidden, axis=0, keepdims=True)
            
            # Update
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            
            loss = np.mean(error**2)
            losses.append(loss)
            if epoch % 400 == 0:
                print(f"[{self.activation.__name__}] Epoch {epoch:4d} → MSE: {loss:.5f}")
        
        return losses

# ────────────────────────────────────────────────
activations = {
    'sigmoid': (sigmoid, lambda x: x*(1-x)),
    'tanh':    (tanh,    lambda x: 1-x**2),
    'relu':    (relu,    lambda x: (x>0).astype(float)),
    'leaky_relu': (lambda x: np.maximum(0.01*x, x), lambda x: np.where(x>0,1,0.01))
}

plt.figure(figsize=(12, 7))

for name, act_fn in activations.items():
    print(f"\nTraining with {name}...")
    net = TinyNet(act_fn[0])
    losses = net.train(X, y, epochs=1500, lr=0.03)
    plt.plot(losses, label=name, linewidth=2, alpha=0.85)

plt.title("Training Loss Comparison - Different Activations\n(small non-linear regression task)", fontsize=13)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()