from functools import reduce
from operator import iconcat

def flatten(a):
    """
    :param numpy.ndarray a: Array to be flattened

    :return: Return a flattened version of a numpy array
    """
    return reduce(iconcat, a, [])
