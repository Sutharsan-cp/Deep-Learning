import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): return z * (1 - z)

class MLP_XOR:
    def __init__(self, lr=0.1):
        np.random.seed(42)
        self.lr = lr
        # Weights: input→hidden, hidden→output
        self.W1 = np.random.randn(2, 2) * 0.1
        self.b1 = np.zeros((1, 2))
        self.W2 = np.random.randn(2, 1) * 0.1
        self.b2 = np.zeros((1, 1))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def train(self, X, y, epochs=20000):
        y = y.reshape(-1, 1)
        for epoch in range(epochs):
            # Forward
            out = self.forward(X)
            
            # Backward
            error2 = (out - y)
            d_out = error2 * sigmoid_deriv(out)
            
            error1 = d_out.dot(self.W2.T)
            d_hidden = error1 * sigmoid_deriv(self.a1)
            
            # Update
            self.W2 -= self.lr * self.a1.T.dot(d_out)
            self.b2 -= self.lr * np.sum(d_out, axis=0, keepdims=True)
            self.W1 -= self.lr * X.T.dot(d_hidden)
            self.b1 -= self.lr * np.sum(d_hidden, axis=0, keepdims=True)
            
            if epoch % 4000 == 0:
                loss = np.mean((out - y)**2)
                print(f"Epoch {epoch:5d} | MSE: {loss:.6f}")

# Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

model = MLP_XOR(lr=0.9)  # higher lr often works better here
model.train(X, y)

# Test
print("\nPredictions:")
for x in X:
    pred = model.forward(x.reshape(1,-1))[0][0]
    print(f"{x} → {pred:.4f}")