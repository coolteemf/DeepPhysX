import numpy as np


def amplitude(m, M, c):
    return c * m + (np.ones(3) - c) * M


def ndim_interpolation(val_min, val_max, count, ignored_dim=None, technic=lambda m, M, c: m + c * M):
    assert (type(val_min) == type(val_max))
    if isinstance(val_min, float) or isinstance(val_min, int):
        return np.array([technic(val_min, val_max, c) for c in np.linspace(0, 1, count)])

    if isinstance(val_min, np.ndarray):
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
                coefficients.append(np.linspace(0, 1, count[i]))

        # Set of all the indices
        valid_idx = {*np.linspace(0, val_min.shape[0] - 1, val_min.shape[0])}
        if ignored_dim is not None:
            # Return the set difference between all the index possible and the ignored ones
            # {0,1,2,3}-{0,2} = {1,3}
            valid_idx = np.array([*(valid_idx - {*ignored_dim})],
                                 dtype=np.int)  # Transform to a list in order to get it as index set

        # The computation starts here
        interp_idx = np.zeros(val_min.shape[0], dtype=np.int)
        values = np.empty((0, *val_min.shape))
        while interp_idx[valid_idx[-1]] < count[-1]:
            coeff = []
            for i in range(val_min.shape[0]):
                coeff.append(coefficients[i][interp_idx[i]])
            values = np.concatenate((values, [technic(val_min, val_max, coeff)]), axis=0)
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