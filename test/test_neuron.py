from pytest import fixture, raises
from neuron import Neuron, UnmatchedInput


@fixture
def three_weights_plus_bias():
    """Returns a Neuron with weights of (-1, 2, -3) and bias of 4"""
    return Neuron([4, -1, 2, -3])


@fixture
def two_inputs():
    """Returns two inputs"""
    return [1, 1]


def test_instantiating_neuron_saves_arguments(three_weights_plus_bias):
    assert three_weights_plus_bias.weights == [4, -1, 2, -3]


def test_weighted_sum_neuron(three_weights_plus_bias):
    assert three_weights_plus_bias.weighted_sum([-1, 0.5, -1/3]) == 7


def test_weighted_sum_inputs_not_matching_weights(
        three_weights_plus_bias, two_inputs):
    assert len(two_inputs) != len(three_weights_plus_bias.weights)
    with raises(UnmatchedInput):
        three_weights_plus_bias.weighted_sum(two_inputs)
