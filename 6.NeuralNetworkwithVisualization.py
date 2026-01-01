import numpy as np

# ==================== NEURON ====================
class Neuron:
    def __init__(self, weights, bias):
        """
        weights : list of numbers (one weight per input)
        bias    : single number
        """
        self.weights = np.array(weights)
        self.bias = bias

    def relu(self, x):
        """
        ReLU activation function:
        f(x) = max(0, x)
        """
        return max(0, x)

    def forward(self, inputs):
        """
        Forward pass of a single neuron
        """
        print("\n    Neuron forward pass")
        print("    -------------------")
        print("    Inputs :", inputs)
        print("    Weights:", self.weights)
        print("    Bias   :", self.bias)

        # Weighted sum (z = w·x + b)
        z = np.dot(self.weights, inputs) + self.bias
        print("    Weighted sum (z = w·x + b):", z)

        # Activation
        output = self.relu(z)
        print("    After ReLU activation:", output)

        return output


# ==================== LAYER ====================
class Layer:
    def __init__(self, neurons_config):
        """
        neurons_config = [
            (weights, bias),
            (weights, bias),
            ...
        ]
        """
        self.neurons = []
        for idx, (weights, bias) in enumerate(neurons_config):
            print(f"Creating Neuron {idx + 1} with weights={weights}, bias={bias}")
            self.neurons.append(Neuron(weights, bias))

    def forward(self, inputs):
        """
        Forward pass through the entire layer
        """
        print("\n  === Layer Forward Pass ===")
        outputs = []

        for i, neuron in enumerate(self.neurons):
            print(f"\n  -> Neuron {i + 1} in this layer")
            neuron_output = neuron.forward(inputs)
            outputs.append(neuron_output)
            print(f"  <- Output from Neuron {i + 1}: {neuron_output}")

        layer_output = np.array(outputs)
        print("\n  Layer output:", layer_output)

        return layer_output


# ==================== NETWORK ====================
class NeuralNetwork:
    def __init__(self, network_config):
        """
        network_config = [
            layer1_config,
            layer2_config,
            ...
        ]
        """
        self.layers = []
        print("\n=== Building Neural Network ===")

        for layer_index, layer_config in enumerate(network_config):
            print(f"\nAdding Layer {layer_index + 1}")
            self.layers.append(Layer(layer_config))

    def forward(self, inputs):
        """
        Forward pass through the whole network
        """
        print("\n==============================")
        print("STARTING NETWORK FORWARD PASS")
        print("==============================")
        print("Initial Input:", inputs)

        for layer_index, layer in enumerate(self.layers):
            print(f"\n>>> Passing through Layer {layer_index + 1}")
            inputs = layer.forward(inputs)

        print("\n==============================")
        print("FINAL NETWORK OUTPUT:", inputs)
        print("==============================")

        return inputs


# ==================== RUN ====================
if __name__ == "__main__":

    # ---------------- MANUAL NETWORK CONFIG ----------------
    network_config = [

        # Hidden Layer (2 neurons, 3 inputs)
        [
            ([0.2, -0.5, 1.0], 0.1),
            ([0.7,  0.3, -0.2], -0.1),
            ([0.7,  0.3, -0.2], -0.1),
            ([0.7,  0.3, -0.2], -0.1),
            ([0.7,  0.3, -0.2], -0.1),
        ],

        # Output Layer (1 neuron, 2 inputs)
        [
            ([1.0, -1.5], 0.2),
        ]
    ]

    # Create network
    nn = NeuralNetwork(network_config)

    # Input vector
    x = np.array([1.0, 0.5, -1.5])

    # Forward pass
    output = nn.forward(x)

    print("\nOutput (Manual Weights):", output)
