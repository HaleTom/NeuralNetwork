import numpy as np


class InsufficientInput(Exception):
    pass


class Neuron:

    def __init__(self, weights, bias=0):
        self.weights = np.array(weights)
        print(weights)
        print(len(self.weights))
        self.bias = bias

    def activate(self, inputs):
        pass

    def weighted_sum(self, inputs):
        if len(inputs) != len(self.weights):
            raise InsufficientInput

        for (value, weight) in zip(inputs, self.weights):
            sum = self.bias
            sum += value * weight
            return sum
