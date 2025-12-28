import numpy as np

weights = np.array([0.5, -0.2, 0.1])
inputs = np.array([1.0, 0.5, -1.5])
bias = 0.4

output = np.dot(weights, inputs) + bias
print("Neuron output:", output)