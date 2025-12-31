import numpy as np

class Neuron:
    def __init__(self, weights):
        self.weights = np.random.rand(weights)
        self.bias = np.random.rand(1)
                
    def relu(self, x):
        return np.maximum(0,x)
    
    def forwardPass(self, inputs):
        return self.relu(np.dot(self.weights, inputs) + self.bias)
    
class Layer:
    def __init__(self, num_neurons, input_size):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]
        
    def forwardPass(self, inputs):
        results = []
        for neuron in self.neurons:
            results.append(neuron.forwardPass(inputs))
        return np.array(results)
    
    
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i+1], layer_sizes[i]))
        
    def forwardPass(self, inputs):
        for layer in self.layers:
            inputs = layer.forwardPass(inputs)
        return inputs
    
# Example usage:
if __name__ == "__main__":
    nn = NeuralNetwork([3,4,2])
    input_data = np.array([1.0, 0.5, -1.5])
    output = nn.forwardPass(input_data)
    print("Network output:", output)