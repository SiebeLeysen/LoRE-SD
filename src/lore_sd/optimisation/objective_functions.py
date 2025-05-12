import numpy as np

def data_fidelity_with_kernel_regularisation(d, S, gaussians, reg_param):
    """
    Compute the objective function as a data fidelity term with kernel regularisation.

    This function calculates the data fidelity term for a given diffusion signal S and a set of Gaussian kernels. It
    involves convolving the orientation distribution function (ODF) with the Gaussian kernels, weighted by the signal
    fractions (fs), and then computing the squared difference between the convolved signal and the actual diffusion
    signal S. A regularization term is added to penalise large values in the kernel to ensure smoothness.

    Parameters:
    - d (np.array): The combined array of ODF and signal fractions fs.
    - S (np.array): The actual diffusion signal.
    - gaussians (np.array): The Gaussian kernels for response function reconstruction.
    - reg_param (float): The regularization parameter controlling the smoothness penalty.

    Returns:
    - float: The objective function, scaled by 1e-4.
    """
    odf, fs = np.split(d, [S.shape[-1]])  # Split d into ODF and fs based on the last dimension of S
    kernel = np.einsum('a, acd -> cd', fs, gaussians)  # Weight the gaussians by fs to get the kernel
    convolved = np.einsum('...ab, ...b -> ...ab', kernel, odf)  # Convolve ODF with the kernel
    differences = (S - convolved)  # Calculate the difference between the actual signal and the convolved signal
    # Compute the data fidelity term with kernel regularization
    return 1e-5*(np.sum(differences ** 2) + reg_param * np.sum(kernel[..., 1:, 1:] ** 2))

def data_fidelity_with_odf_regularisation(d, S, gaussians, Q, reg_param):
    """
    Compute the objective function as a data fidelity term with kernel regularisation.

    This function calculates the data fidelity term for a given diffusion signal S and a set of Gaussian kernels. It
    involves convolving the orientation distribution function (ODF) with the Gaussian kernels, weighted by the signal
    fractions (fs), and then computing the squared difference between the convolved signal and the actual diffusion
    signal S. A regularization term is added to penalise large values in the kernel to ensure smoothness.

    Parameters:
    - d (np.array): The combined array of ODF and signal fractions fs.
    - S (np.array): The actual diffusion signal.
    - gaussians (np.array): The Gaussian kernels for response function reconstruction.
    - reg_param (float): The regularization parameter controlling the smoothness penalty.

    Returns:
    - float: The objective function, scaled by 1e-4.
    """
    odf, fs = np.split(d, [S.shape[-1]])  # Split d into ODF and fs based on the last dimension of S
    kernel = np.einsum('a, acd -> cd', fs, gaussians)  # Weight the gaussians by fs to get the kernel
    convolved = np.einsum('...ab, ...b -> ...ab', kernel, odf)  # Convolve ODF with the kernel
    differences = (S - convolved)  # Calculate the difference between the actual signal and the convolved signal
    # Compute the data fidelity term with kernel regularization
    return 1e-5*(np.sum(differences ** 2) + reg_param * np.linalg.norm((Q @ odf).flatten(), ord=.2))


def jac_data_fidelity_with_kernel_regularisation(d, S, gaussians, reg_param):
    """
    Compute the jacobian of the objective function.

    Parameters:
    - d (np.array): The combined array of ODF and signal fractions fs.
    - S (np.array): The actual diffusion signal.
    - gaussians (np.array): The Gaussian kernels for response function reconstruction.
    - reg_param (float): The regularization parameter controlling the smoothness penalty.

    Returns:
    - np.array: The jacobion of the objective function, scaled by 1e-4.
    """
    nlmax = S.shape[-1]
    odf, fs = np.split(d, [nlmax])
    kernel = np.einsum('a, acd -> cd', fs, gaussians)
    convolved = np.einsum('...ab, ...b -> ...ab', kernel, odf)
    differences = S - convolved  # Calculate differences directly
    grad = np.zeros_like(d)
    grad[:nlmax] = -2 * np.sum(differences * kernel, axis=0)

    grad[nlmax:] = -2 * np.einsum('...ab, ...ab -> ...', differences, gaussians * odf)
    grad[nlmax:] += reg_param * 2 * np.einsum('...ij, ...ij -> ...', kernel[..., 1:, 1:], gaussians[..., 1:, 1:])

    return 1e-5 * grad


def csd_fit(odf, S, rf):
    convolved = np.einsum('...ab, ...b -> ...ab', rf, odf)
    differences = S - convolved  # Calculate differences directly
    return 1e-4 * np.sum(differences ** 2)

def jac_csd_fit(odf, S, rf):
    convolved = np.einsum('...ab, ...b -> ...ab', rf, odf)
    differences = S - convolved  # Calculate differences directly
    grad = -2 * np.sum(differences * rf, axis=0)
    return 1e-4 * grad
