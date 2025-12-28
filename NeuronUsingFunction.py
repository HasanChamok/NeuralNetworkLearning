import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.random.rand(weights)
        self.bias = bias
        # self.bias = np.random.randn() #takes a random value standard normal distribution whereas rand takes uniform distribution
        #np.random.rand() range[0.0, 1.0) whereas np.random.randn() range(-inf, +inf)
        
    def forwarPass(self, inputs):
        return np.dot(self.weights, inputs) + self.bias
    

Network = Neuron(3, 0.5)
inputs = np.array([1.0, 0.5, -1.5])
output = Network.forwarPass(inputs)
print("Neuron output:", output) 

    