import numpy as np
import activation_functions as activations


class InsufficientInput(Exception):
    pass


class Neuron:

    def __init__(self, weights, bias=0, activation_func=activations.sigmoid):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation_fn = activation_func

    def activate(self, inputs):
        return self.sigmoid(self, self.weighted_sum(inputs))

    def sigmoid(self, value):
        return value

    def weighted_sum(self, inputs):
        if len(inputs) != len(self.weights):
            raise InsufficientInput

        for (value, weight) in zip(inputs, self.weights):
            sum = self.bias
            sum += value * weight
            return sum
