from activation_functions import sigmoid
from neuron import Neuron


class Layer:

    def __init__(self, num_neurons, prev=None, next=None, activation_fn=sigmoid):
        self.neurons = [Neuron() for each in range(num_neurons)]
        self.prev = prev
        self.next = next
        self.activation_fn = activation_fn

    def __len__(self):
        """Number of neurons in this layer"""
        return len(self.neurons)

    def output(self, inputs):
        return [self.activation_fn(n.weighted_sum(inputs)) for n in self.neurons]

    @property
    def weights(self):
        """The weights of all the neurons"""
        return [n.weights for n in self.neurons]

    @weights.setter
    def weights(self, weights_matrix):
        if len(weights_matrix) != len(self.neurons):
            raise IndexError("Weights don't fit neurons")

        for weights, neuron in zip(weights_matrix, self.neurons):
            neuron.weights = weights
