import numpy as np
import matplotlib.pyplot as plt

# =============================================
# 1. Generate NON-LINEAR toy data
# =============================================
np.random.seed(42)
n_samples = 200
X = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)

# True non-linear target function
y_true = np.sin(1.5 * X) + 0.4 * X**2
y = y_true + np.random.normal(0, 0.18, X.shape)  # small noise

# Sort for nice plotting
sort_idx = np.argsort(X.ravel())
X_plot = X[sort_idx]
y_plot = y[sort_idx]

# =============================================
# 2. Activation functions
# =============================================
def sigmoid(z):    return 1 / (1 + np.exp(-z))
def tanh(z):       return np.tanh(z)
def relu(z):       return np.maximum(0, z)
def leaky_relu(z, alpha=0.01): return np.where(z >= 0, z, alpha * z)

# =============================================
# 3. Very simple 1-hidden-layer network
# =============================================
class TinyMLP:
    def __init__(self, activation):
        self.activation = activation
        self.act_deriv = self._get_deriv()
        np.random.seed(42)
        self.W1 = np.random.randn(1, 12) * 0.4     # 12 hidden neurons
        self.b1 = np.zeros((1, 12))
        self.W2 = np.random.randn(12, 1) * 0.4
        self.b2 = np.zeros((1, 1))
    
    def _get_deriv(self):
        if self.activation == sigmoid:
            return lambda a: a * (1 - a)
        elif self.activation == tanh:
            return lambda a: 1 - a**2
        elif self.activation == relu:
            return lambda a: (a > 0).astype(float)
        elif self.activation == leaky_relu:
            return lambda a: np.where(a >= 0, 1.0, 0.01)
        return lambda a: a  # fallback
    
    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2
    
    def train(self, X, y, lr=0.03, epochs=1800):
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
            
            mse = np.mean(error**2)
            losses.append(mse)
            
            if epoch % 600 == 0:
                print(f"[{self.activation.__name__}] Epoch {epoch:4d} | MSE: {mse:.5f}")
                
        return losses, self.forward(X_plot)  # final prediction for plotting

# =============================================
# 4. Train with different activations
# =============================================
activations = {
    'sigmoid': sigmoid,
    'tanh':    tanh,
    'relu':    relu,
    'leaky_relu': leaky_relu
}

plt.figure(figsize=(14, 9))

# Plot real data
plt.scatter(X, y, s=35, alpha=0.5, color='gray', label='noisy data')

for name, act_fn in activations.items():
    print(f"\nTraining with {name}...")
    model = TinyMLP(act_fn)
    losses, y_pred = model.train(X, y, epochs=1800, lr=0.035)
    
    # Plot final fit
    plt.plot(X_plot, y_pred, label=f'{name} fit', linewidth=2.4, alpha=0.9)
    
    # Optional: plot loss curve in inset if you want
    # (commented out to keep main plot clean)

plt.plot(X_plot, y_true[sort_idx], 'k--', lw=1.8, alpha=0.7, label='true function')
plt.title("Which activation helps best learn non-linear pattern?\n(sin(1.5x) + 0.4x²)", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()