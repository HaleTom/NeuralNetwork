def is_close(first, second, allowable_difference):
    """True IFF the difference between two values is allowable"""
    return abs(first - second) <= allowable_difference
