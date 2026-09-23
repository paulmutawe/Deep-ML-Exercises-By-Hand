import numpy as np

def mini_batch_gd_step(X, y, weights, bias, batch_indices, lr):
    
    batch_X = X[batch_indices]
    batch_y = y[batch_indices]
    
    predictions = batch_X @ weights + bias
    
    errors = predictions - batch_y
    
    batch_size = len(batch_indices)
    
    weight_gradient =  (2/batch_size) * (batch_X.T @ errors)
    bias_gradient = (2/batch_size) * np.sum(errors)
    
    weights = weights - lr * weight_gradient
    bias = bias - lr * bias_gradient
    
    return np.append(weights, bias)


X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [0, 1]
])

y = np.array([5, 4, 11, 2])

weights = np.array([0.0, 0.0])
bias = 0.0

batch_indices = [0, 1]
lr = 0.1


print(mini_batch_gd_step(
    X,
    y,
    weights,
    bias,
    batch_indices,
    lr
))
