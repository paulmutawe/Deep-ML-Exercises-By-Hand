import numpy as np

def momentum_optimizer(parameter, grad, velocity, leanring_rate=0.01, momentum=0.9):
    
    updated_velocity = momentum * velocity + leanring_rate * grad
    
    updated_parameter = parameter - updated_velocity
    
    return np.round(updated_parameter, 5), np.round(updated_velocity, 5) 

parameter = 1.0
grad = 0.1
velocity = 0.1

print(momentum_optimizer(parameter, grad, velocity))
