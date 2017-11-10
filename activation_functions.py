from math import pow, e


def sigmoid(x):
    """Return the sigmoid of x"""
    return 1 / (1 + pow(e, -x))
