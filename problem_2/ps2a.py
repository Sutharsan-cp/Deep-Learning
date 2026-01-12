import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def predict(self, X):
        return 1 if np.dot(X, self.weights) + self.bias > 0 else 0
    
    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                prediction = 1 if np.dot(x_i, self.weights) + self.bias > 0 else 0
                update = self.lr * (y[idx] - prediction)
                self.weights += update * x_i
                self.bias += update

# Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

# Train
model = Perceptron(lr=0.5, epochs=12)
model.train(X, y)

# Test
print("Predictions:")
for x, target in zip(X, y):
    pred = model.predict(x)
    print(f"{x} → {pred}  (target: {target})")