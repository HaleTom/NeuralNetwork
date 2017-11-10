class Layer:

    def __init__(self, neurons, prev=None, next=None):
        self.neurons = [None] * neurons
        self.prev = prev
        self.next = next
