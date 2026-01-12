import numpy as np
import matplotlib.pyplot as plt

def beale(x, y):
    term1 = (1.5 - x + x * y)**2
    term2 = (2.25 - x + x * y**2)**2
    term3 = (2.625 - x + x * y**3)**2
    return term1 + term2 + term3

def beale_gradient(x, y):
    # Partial derivatives computed symbolically (you can derive them)
    dx = 2*(1.5 - x + x*y)*(y - 1) + \
         2*(2.25 - x + x*y**2)*(y**2 - 1) + \
         2*(2.625 - x + x*y**3)*(y**3 - 1)
    
    dy = 2*(1.5 - x + x*y)*x + \
         2*(2.25 - x + x*y**2)*(2*x*y) + \
         2*(2.625 - x + x*y**3)*(3*x*y**2)
    
    return np.array([dx, dy])

# Gradient Descent
np.random.seed(42)
params = np.array([0.0, 0.0])          # starting point (try others: [1,1], [-2,2], etc.)
lr = 0.001
max_iter = 20000
history = []

for i in range(max_iter):
    grad = beale_gradient(*params)
    params -= lr * grad
    loss = beale(*params)
    history.append(loss)
    
    if i % 2000 == 0:
        print(f"Iter {i:5d} | loss = {loss:.8f} | x,y = {params}")
    
    if loss < 1e-8:
        print("Reached near-global minimum!")
        break

print("\nFinal solution:")
print(f"x, y   = {params}")
print(f"f(x,y) = {beale(*params):.10f}")