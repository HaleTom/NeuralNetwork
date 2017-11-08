import pytest
from neuron import Neuron, InsufficientInput


@pytest.fixture
def three_weights_plus_bias():
    """Returns a Neuron with weights of (-1, 2, -3) and bias of 4"""
    return Neuron([-1, 2, -3], 4)


@pytest.fixture
def two_weights():
    """Returns two weights"""
    return [1, 1]


def test_instantiating_neuron_saves_weights_and_bias(three_weights_plus_bias):
    assert three_weights_plus_bias.bias == 4
    assert three_weights_plus_bias.weights.tolist() == [-1, 2, -3]


def test_weighted_sum_neuron(three_weights_plus_bias):
    assert three_weights_plus_bias.weighted_sum([-1, 0.5, -1/3]) == 5


def test_weighted_sum_inputs_not_matching_weights(
        three_weights_plus_bias, two_weights):
    assert len(two_weights) != len(three_weights_plus_bias.weights.tolist())
    with pytest.raises(InsufficientInput):
        three_weights_plus_bias.weighted_sum(two_weights)
