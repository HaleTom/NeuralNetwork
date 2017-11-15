from layer import Layer
from activation_functions import sigmoid


class InvalidArchitecture(Exception):
    """There must be at least two layers in a Neural Network"""


class Network:

    def __init__(self, neurons_per_layer, activation_fn=sigmoid):
        if len(neurons_per_layer) < 2:
            raise InvalidArchitecture("There must be more than one layer")

        # Add layers to list
        self.layers = []
        for neurons in neurons_per_layer:
            self.layers.append(Layer(neurons, activation_fn=activation_fn))

        # link layers
        prev = self.layers[0]
        for layer in self.layers[1:]:
            layer.prev = prev
            prev.next = layer
            prev = layer  # Get ready to do it again

    def output(self, input):
        for layer in self.layers:
            out = layer.output(input)
            input = out
        return out
