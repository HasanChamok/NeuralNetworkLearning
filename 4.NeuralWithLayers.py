import numpy as np


class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.random.rand(weights)
        self.bias = bias
        
    def relu(self, x):
        return max(0, x)
    
    def forwardPass(self, inputs):
        return self.relu(np.dot(self.weights, inputs) + self.bias)
    

class Layer:
    def __init__(self, num_neurons, input_size):
        self.neurons = [Neuron(input_size, np.random.randn()) for _ in range(num_neurons)]
        
    def forwardPass(self, inputs):
        return np.array([neuron.forwardPass(inputs) for neuron in self.neurons])
    

network = Layer(3, 3)
inputs = np.array([1.0, 0.5, -1.5])
output = network.forwardPass(inputs)
print("Layer output after ReLU activation:", output)