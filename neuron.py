import numpy as np
from functools import reduce


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

        return reduce(lambda x, y: x + y[0] * y[1], zip(inputs, self.weights), 0)
