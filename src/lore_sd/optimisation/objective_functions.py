import numpy as np
import numba

@numba.njit(fastmath=True)
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
    
    # Weight the gaussians by fs to get the kernel
    kernel = np.zeros((gaussians.shape[1], gaussians.shape[2]))
    for i in range(fs.shape[0]):
        kernel += fs[i] * gaussians[i]
    
    # Convolve ODF with the kernel
    convolved = np.zeros_like(S)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            convolved[i, j] = kernel[i, j] * odf[j]
    
    differences = S - convolved  # Calculate the difference between the actual signal and the convolved signal
    
    # Compute the data fidelity term with kernel regularization
    data_fidelity = np.sum(differences ** 2)
    kernel_regularization = np.sum(kernel[1:, 1:] ** 2)
    
    return 1e-5 * (data_fidelity + reg_param * kernel_regularization)
    # odf, fs = np.split(d, [S.shape[-1]])  # Split d into ODF and fs based on the last dimension of S
    # kernel = np.einsum('a, acd -> cd', fs, gaussians)  # Weight the gaussians by fs to get the kernel
    # convolved = np.einsum('...ab, ...b -> ...ab', kernel, odf)  # Convolve ODF with the kernel
    # differences = (S - convolved)  # Calculate the difference between the actual signal and the convolved signal
    # # Compute the data fidelity term with kernel regularization
    # return 1e-5*(np.sum(differences ** 2) + reg_param * np.sum(kernel[..., 1:, 1:] ** 2))

@numba.njit(fastmath=True)
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
    
    # Compute kernel
    kernel = np.zeros((gaussians.shape[1], gaussians.shape[2]))
    for i in range(fs.shape[0]):
        for j in range(gaussians.shape[1]):
            for k in range(gaussians.shape[2]):
                kernel[j, k] += fs[i] * gaussians[i, j, k]
    
    # Compute convolved
    convolved = np.zeros_like(S)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            convolved[i, j] = kernel[i, j] * odf[j]
    
    differences = S - convolved  # Calculate differences directly
    
    grad = np.zeros_like(d)
    
    # Compute grad[:nlmax]
    for i in range(nlmax):
        numba.numba_print((-2 * np.sum(differences * kernel[:,i], axis=0)).shape)
        grad[i] = -2 * np.sum(differences * kernel[:,i], axis=0)
    
    # Compute grad[nlmax:]
    for i in range(fs.shape[0]):
        grad[nlmax + i] = -2 * np.sum(differences * gaussians[i] * odf, axis=(0, 1))
    
    # Add regularization term to grad[nlmax:]
    for i in range(1, kernel.shape[0]):
        for j in range(1, kernel.shape[1]):
            grad[nlmax:] += reg_param * 2 * kernel[i, j] * gaussians[:, i, j]
    
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
