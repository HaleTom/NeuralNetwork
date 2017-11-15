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

        accumulator = 0
        for (value, weight) in zip(inputs, self.weights):
            accumulator += value * weight
        return accumulator
