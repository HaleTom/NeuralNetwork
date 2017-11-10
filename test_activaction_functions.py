import activation_functions as a
from test_helper import *


def test_sigmoid_activation():
    """Test some points of the sigmoid function"""
    assert is_close(a.sigmoid(0), 0.5, 1e-9)
    assert is_close(a.sigmoid(5), 0.993307149076, 1e-9)
    assert is_close(a.sigmoid(-1), 0.26894142137, 1e-9)
