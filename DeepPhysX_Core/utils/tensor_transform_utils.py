import functools
import operator


def flatten(a):
    """
    :param numpy.ndarray a: Array to be flattened

    :return: Return a flattened version of a numpy array
    """
    return functools.reduce(operator.iconcat, a, [])
