import activation_functions as a
import numpy as np


def test_sigmoid_activation():
    """Test some points of the sigmoid function"""
    assert np.isclose(a.sigmoid(0), 0.5, 1e-9)
    assert np.isclose(a.sigmoid(5), 0.993307149076, 1e-9)
    assert np.isclose(a.sigmoid(-1), 0.26894142137, 1e-9)
