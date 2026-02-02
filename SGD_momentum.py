import torch

x = torch.tensor([-2.8], requires_grad=True)  # starting point

# objective function
def objective(x):
    return 0.3 * x**4 - 0.1 * x**3 - 2 * x**2 - 0.8 * x

# SGD optimizer with momentum
optimizer = torch.optim.SGD([x], lr=0.05, momentum=0.9)

# Training loop
for step in range(1000):
    optimizer.zero_grad()
    loss = objective(x)
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"Step {step}: x = {x.item():.4f}, loss = {loss.item():.4f}")
