import numpy as np

class NeuralNetwork:
    def __init__(self, weights, bias):
        self.weights = np.random.rand(weights)
        self.bias = bias
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward_pass(self, inputs):
        return self.relu(np.dot(self.weights,  inputs)+self.bias)
    
network = NeuralNetwork(3, 0.5)
inputs = np.array([1.0, 0.5, -1.5])
output = network.forward_pass(inputs)
print("Neuron output after ReLU activation:", output)
