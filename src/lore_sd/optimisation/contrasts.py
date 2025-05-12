import numpy as np

def normalise_0_1(vals):
    return (vals - np.min(vals)) / (np.max(vals) - np.min(vals))

def free_water_contrast(ad_range, rd_range):
    """
    Calculate the free water contrast matrix.

    This function generates a matrix representing the contrast for free water based on axial diffusivity (AD) and
    radial diffusivity (RD) ranges. The contrast is set to 1 if the conditions 1000 * AD >= 2.6, 1000 * RD >= 2.6,
    and AD >= RD are met, indicating the presence of free water.

    Parameters:
    - ad_range (list or np.array): A list or array of AD values.
    - rd_range (list or np.array): A list or array of RD values.

    Returns:
    - np.array: A 2D numpy array representing the free water contrast.
    """
    out = np.zeros((len(ad_range), len(rd_range)))

    for i, ad in enumerate(ad_range):
        for j, rd in enumerate(rd_range):
            if 1000 * ad >= 2.6 and 1000 * rd >= 2.6 and ad >= rd:
                out[i, j] = 1

    return out

def exponential_decay_function(vals, reverse=False, rate=10):
    if not reverse:
        return normalise_0_1(np.array([np.exp(-rate * j) for j in normalise_0_1(vals)]))
    else:
        return normalise_0_1(np.array([np.exp(-rate * (vals[-1] - j)) for j in normalise_0_1(vals)]))

def to_decay_matrix(ad, rd, decay_function, axis='radial', with_isotropic=True, **kwargs):
    unit_matrix = ad[:, None] >= rd[None] if with_isotropic else ad[:, None] > rd[None]
    if axis == 'radial':
        dec = decay_function(rd, **kwargs)
        dec = np.repeat(dec[None], repeats=len(ad), axis=0)
    elif axis == 'axial':
        dec = decay_function(ad, **kwargs)
        dec = np.repeat(dec[:, None], repeats=len(rd), axis=1)
    else:
        raise ValueError('Axis must be either "radial" or "axial"')
    return unit_matrix * dec


def intra_axonal_contrast(ad_range, rd_range, with_isotropic=True, rate=10):
    dec_matrix = to_decay_matrix(ad_range, rd_range, exponential_decay_function, axis='radial', with_isotropic=with_isotropic, rate=rate)
    return dec_matrix

def rfa_map(ad, rd):

    ad_matrix = np.repeat(ad[..., None], len(rd), axis=-1)
    rd_matrix = np.repeat(rd[None, ...], len(ad), axis=0)

    mask = ad_matrix >= rd_matrix
    ad_matrix *= mask
    rd_matrix *= mask

    lambda_mean = (ad_matrix + 2*rd_matrix) / 3
    return np.nan_to_num(np.sqrt(3/2) * np.sqrt((ad_matrix - lambda_mean)**2 + (rd_matrix - lambda_mean)**2 + (rd_matrix - lambda_mean)**2) / np.sqrt(ad_matrix**2 + rd_matrix**2 + rd_matrix**2))


def extra_axonal_contrast(ad_range, rd_range, with_isotropic=True, rate=10):
    """
    Calculate the extra-axonal contrast matrix.

    This function generates a matrix representing the contrast for extra-axonal spaces based on axial diffusivity (
    AD) and radial diffusivity (RD) ranges. It is calculated as the complement to the sum of free water and
    intra-axonal contrasts, effectively representing the remaining contrast space.

    Parameters:
    - ad_range (list or np.array): A list or array of AD values.
    - rd_range (list or np.array): A list or array of RD values.

    Returns:
    - np.array: A 2D numpy array representing the extra-axonal contrast.
    """
    out = 1 - free_water_contrast(ad_range, rd_range) - intra_axonal_contrast(ad_range, rd_range, with_isotropic=with_isotropic, rate=rate)
    out[ad_range[:, None] < rd_range[None, :]] = 0
    return out

def get_contrast(fs, ad, rd, weighting_function, *args):
    """
    Calculate the weighted contrast based on a given weighting function.

    This function computes the contrast by applying a weighting function to the axial diffusivity (AD) and radial
    diffusivity (RD) values, then multiplying the resulting weights with the signal fractions (fs) and summing over
    the last two dimensions.

    Parameters:
    - fs (np.array): A numpy array representing the signal fractions.
    - ad (np.array): A numpy array of axial diffusivity values.
    - rd (np.array): A numpy array of radial diffusivity values.
    - weighting_function (function): A function that takes AD and RD and returns a weight matrix.

    Returns:
    - float: The sum of the weighted contrast over the specified axes.
    """
    # Calculate weights using the provided weighting function
    weights = weighting_function(ad, rd, *args)

    # Return the sum of the product of weights and signal fractions over the last two dimensions
    return np.sum(weights * fs, axis=(-1, -2))
