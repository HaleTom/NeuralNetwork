from pytest import fixture
from layer import Layer

# Note that these fixtures can't be fully linked due to recursion.
# See issue I raised when discovering this:
# https://github.com/pytest-dev/pytest/issues/2909


@fixture
def input_layer():
    """Linked input layer with 2 neurons"""
    return Layer(2, prev=None)


@fixture
def hidden_layer(input_layer, output_layer):
    """Linked hidden layer with 3 neurons"""
    return Layer(3, prev=input_layer, next=output_layer)


@fixture
def output_layer():
    """Linked output layer with a single neuron"""
    return Layer(3, next=None)


def test_init_creates_list_of_neurons():
    layer = Layer(3)
    assert len(layer.neurons) == 3


def test_init_saves_given_parameters(input_layer, hidden_layer, output_layer):
    assert hidden_layer.prev is input_layer
    assert hidden_layer.next is output_layer
