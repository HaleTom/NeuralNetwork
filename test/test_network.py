import pytest
from network import Network, InvalidArchitecture


@pytest.fixture
def two_three_one_network():
    return Network((2, 3, 1))


@pytest.fixture
def xor_network():
    return Network(3, 1)


def test_specified_neurons_exist_per_layer(two_three_one_network):
    n = two_three_one_network
    assert len(n.layers[0]) == 2
    assert len(n.layers[1]) == 3
    assert len(n.layers[2]) == 1


def test_layer_forward_linkage(two_three_one_network):
    n = two_three_one_network
    assert n.layers[0].next is n.layers[1]
    assert n.layers[1].next is n.layers[2]
    assert n.layers[2].next is None


def test_layer_backward_linkage(two_three_one_network):
    n = two_three_one_network
    assert n.layers[0].prev is None
    assert n.layers[1].prev is n.layers[0]
    assert n.layers[2].prev is n.layers[1]


def test_layers_must_be_greater_than_one():
    with pytest.raises(InvalidArchitecture):
        Network([3])


def test_network_output_with_identity_activation_function():
    def fn(x): return x
    n = Network((3, 1), activation_fn=lambda x: x)
    n.layers[0].weights = ((3, 1, 2),
                           (0, -1, 0.5),
                           (1, -1, 0.5))
    # n.layers[0].weights = ((2, 3, 1),
    #                        (.5, 0, -1),
    #                        (.5, 1, -1))
    n.layers[1].weights = ((.5, .5, 100, 4),)
    assert n.output([2, 4])[0] == 11
