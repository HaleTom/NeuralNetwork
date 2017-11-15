import pytest
from layer import Layer
from neuron import Neuron
from activation_functions import sigmoid

# Note that these fixtures can't be fully linked due to recursion.
# See issue I raised when discovering this:
# https://github.com/pytest-dev/pytest/issues/2909


@pytest.fixture
def input_layer():
    """Linked input layer with 2 neurons"""
    return Layer(2, prev=None)


@pytest.fixture
def hidden_layer(input_layer, output_layer):
    """Linked hidden layer with 3 neurons"""
    return Layer(3, prev=input_layer, next=output_layer)


@pytest.fixture
def output_layer():
    """Linked output layer with a single neuron"""
    return Layer(3, next=None)


def test_init_creates_list_of_neurons():
    layer = Layer(3)
    assert len(layer.neurons) == 3


def test_init_links_adjacent_layers(input_layer, hidden_layer, output_layer):
    assert hidden_layer.prev is input_layer
    assert hidden_layer.next is output_layer


def test_init_stores_activation_function():
    def fn(): pass
    assert (Layer(1, activation_fn=fn).activation_fn == fn)


def test_len_returns_number_of_neurons(output_layer):
    assert len(output_layer) == 3


def test_weights_must_fit_neurons(hidden_layer):
    with pytest.raises(IndexError):
        hidden_layer.weights = [[1], [2]]


def test_weights_can_be_set_and_got(hidden_layer):
    hidden_layer.weights = [[1, 2], [2, 3], [3, 4]]
    assert hidden_layer.weights == [[1, 2], [2, 3], [3, 4]]


def test_activation_function(hidden_layer):
    hidden_layer.weights = ([[3, -1, 2, -3],
                             [4, -1, 2, -3],
                             [5, -1, 2, -6]])
    assert hidden_layer.output([-1, 0.5, -1/3]) == \
        [sigmoid(6), sigmoid(7), sigmoid(9)]
