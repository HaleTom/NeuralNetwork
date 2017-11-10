import numpy as np
import activation_functions as activations
from functools import lru_cache


class InsufficientInput(Exception):
    pass


class Neuron:

    def __init__(self, weights, bias=0, activation_func=activations.sigmoid):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation_fn = activation_func

    @lru_cache(8)
    def output(self, inputs):
        self.output = self.activation_fn(self.weighted_sum(inputs))
        return self.output

    def sigmoid(self, value):
        return value

    def weighted_sum(self, inputs):
        if len(inputs) != len(self.weights):
            raise InsufficientInput

        for (value, weight) in zip(inputs, self.weights):
            sum = self.bias
            sum += value * weight
            return sum
