from math import exp


def sigmoid(x):
    """Return the sigmoid of x"""
    # pi says exp(x) is more efficient
    # If you go through my differentiation video you will learn why:
    # a pretty power series expansion exp(x) = 1 + x + x^2/2! + x^3/3! + ...
    return 1 / (1 + exp(-x))
