# 1) Diseñar el modelo (input, output size, forward pass)
# 2) Contruir la funcion de perdida y optimizador
# 3) Training loop
#   - forward pass: compute prediction and loss
#   - backward pass: compute gradients
#   - update weights
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, 
                                            n_features=1, 
                                            noise=20, 
                                            random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) model
input_size = n_features # 1
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) loss y optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_predicted = model(X)
    loss = criterion(y_predicted, y)    
    
    # backward pass
    loss.backward() # gradients

    # update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch: {epoch+1}, loss = {loss.item():.4}, weights = {w[0][0].item()}, bias = {b[0].item()}')

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()