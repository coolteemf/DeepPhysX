import math
from numpy import array, concatenate, ndarray, append, zeros, empty
from numpy import cos, sin, sqrt, exp, pi, linspace
from torch import tensor


def nextPowerOf2(n):
    # if n = 0 or is a power of 2 return n
    if n and not (n & (n - 1)):
        return n
    return 1 << (int(math.log(n)) + 1)


def fibonacci3DSphereSampling(ng, step, shift_phi=0, shift_theta=0):
    phi = (1.0 + sqrt(5.0)) / 2.0

    theta = array([])
    sphi = array([])
    cphi = array([])

    for i in range(0, ng, step):
        i2 = 2.0 * i - (ng - 1.0)
        theta = append(theta, 2.0 * pi * float((i2 + shift_theta) % ng) / phi)
        sphi = append(sphi, float(i2 + shift_phi) / float(ng))
        cphi = append(cphi, sqrt(float(ng + i2 + shift_phi) * float(ng - i2)) / float(ng))

    tmp_xg = list(filter(lambda x: x[0] >= 0.0, [[cphi[i] * sin(theta[i]), cphi[i] * cos(theta[i]), sphi[i]] for i in range(len(sphi))]))
    xg = tensor(tmp_xg)

    return xg


def sigmoid(x, s=0, k=20):
    return 1 / (1 + exp(-k * x + s))


def min_max_feature_scaling(feature, min, max, epsilon=0):
    return (feature - min + epsilon) / (max - min + epsilon)


def ndim_interpolation(val_min, val_max, count, ignored_dim=None, technic=lambda m, M, c: m + c * M):
    assert (type(val_min) == type(val_max))
    if isinstance(val_min, float) or isinstance(val_min, int):
        return array([technic(val_min, val_max, c) for c in linspace(0, 1, count)])

    if isinstance(val_min, ndarray):
        # if count is an int make it a list of val_min dim of itself
        #  Ex : count = 10 , val_min = [14,-8,1] -> count = [10, 10, 10]
        if isinstance(count, int):
            count = [count]*val_min.shape[0]
        # Coefficient is a list of array with the corresponding subdivision
        coefficients = []
        for i in range(val_min.shape[0]):
            if count[i] <= 0:
                coefficients.append([0.0])
            else:
                coefficients.append(linspace(0, 1, count[i]))

        # Set of all the indices
        valid_idx = {*linspace(0, val_min.shape[0] - 1, val_min.shape[0])}
        if ignored_dim is not None:
            # Return the set difference between all the index possible and the ignored ones
            # {0,1,2,3}-{0,2} = {1,3}
            valid_idx = array([*(valid_idx - {*ignored_dim})],
                                 dtype=int)  # Transform to a list in order to get it as index set

        # The computation starts here
        interp_idx = zeros(val_min.shape[0], dtype=int)
        values = empty((0, *val_min.shape))
        while interp_idx[valid_idx[-1]] < count[-1]:
            coeff = []
            for i in range(val_min.shape[0]):
                coeff.append(coefficients[i][interp_idx[i]])
            values = concatenate((values, [technic(val_min, val_max, coeff)]), axis=0)
            interp_idx[valid_idx[0]] += 1
            # Check if we reached the number of subdivision for the dimension
            # if so we carry the extra value
            i = 0
            while interp_idx[valid_idx[i]] >= count[valid_idx[i]] and i < len(valid_idx)-1:
                interp_idx[valid_idx[i]] = 0
                interp_idx[valid_idx[i + 1]] += 1
                i += 1
        return values
    return None