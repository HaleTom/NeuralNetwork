import numpy as np


class UnmatchedInput(Exception):
    pass


class Neuron:

    def __init__(self, weights=None):
        """Weights includes the bias as the first element"""
        self.weights = weights

    def weighted_sum(self, inputs):
        inputs = [1, *inputs]
        if len(inputs) != len(self.weights):
            raise UnmatchedInput

        return np.dot(inputs, self.weights)
