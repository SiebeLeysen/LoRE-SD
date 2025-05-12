import multiprocessing as mp
import os

from scipy.optimize import minimize

import numpy as np
import time

from lore_sd.utils import SphericalHarmonics as sh
from lore_sd.optimisation import objective_functions
from lore_sd.optimisation import constraints
from lore_sd.utils import io_utils, math_utils, gradient_utils

import subprocess
import tqdm

def get_signal_decomposition(dwi, mask, grad, Da, Dr, reg=1e-3, Q=None, lmax=8, cores=50):
    """
    Perform blind deconvolution on diffusion-weighted imaging (DWI) data.

    This function applies a blind deconvolution algorithm to estimate orientation distribution functions (ODFs),
    response functions, and response function composition from DWI data.

    Parameters:
    - dwi (numpy.ndarray): The DWI data as a 4D array where the first three dimensions are spatial and the last one is the gradient direction.
    - mask (numpy.ndarray): A 3D boolean array indicating the brain mask. Only voxels within the mask are processed.
    - grad (numpy.ndarray): The gradient table including b-values and directions. The last column should contain the b-values.
    - Da (list or numpy.ndarray): Axial diffusivities for the response function estimation.
    - Dr (list or numpy.ndarray): Radial diffusivities for the response function estimation.
    - reg (float): Regularisation parameter used in the optimization.
    - lmax (int, optional): Maximum spherical harmonics order. Defaults to 8.
    - cores (int, optional): Number of cores to use for multiprocessing. Defaults to 50.

    Returns:
    dict: A dictionary containing:
        - 'odf': Estimated orientation distribution functions as a 4D array.
        - 'response': Estimated response functions as a 3D array.
        - 'fs': Estimated fiber fractions as a 3D array.
        - Execution time is printed to the console.

    """
    start_time = time.time()
    grad[..., -1] = np.round(grad[..., -1], -2)  # Round b-values to nearest 100
    bvals = np.unique(grad[..., -1])  # Extract unique b-values
    M = len(bvals)  # Number of unique b-values
    mask_len = np.count_nonzero(mask)  # Number of voxels within the mask
    obj_fun = objective_functions.data_fidelity_with_kernel_regularisation  # Objective function
    jac = objective_functions.jac_data_fidelity_with_kernel_regularisation  # Jacobian of the objective function
    
    if Q is None:
        Q = get_transformation_matrix(600, lmax)  # Transformation matrix

    # Prepare arguments for multiprocessing
    # Arguments used for optimisation of voxel
    args = [(voxel, Da, Dr, grad, lmax, reg, Q, obj_fun, jac) for voxel in
            dwi[mask]]
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    pool = mp.Pool(cores)

    results = pool.starmap(decompose_voxel, tqdm.tqdm(args, total=len(args)), chunksize=1)

    pool.close()
    pool.join()

    # Initialize arrays for ODFs, responses, and fiber fractions
    odfs, init_odfs, responses, fs = map(np.zeros,
                              [(mask_len, sh.n4l(lmax)), (mask_len, sh.n4l(lmax)), (mask_len, M, lmax // 2 + 1), (mask_len, len(Da), len(Dr))])
    for i, result in enumerate(results):
        odfs[i], init_odfs[i], responses[i], fs[i] = result['odf'], result['init_odf'], result['response'], result['gaussian_fractions']

    # Simplified version of creating output arrays with the correct shape
    shapes = [
        mask.shape + (sh.n4l(lmax),),
        mask.shape + (sh.n4l(lmax),),
        mask.shape + (M, lmax // 2 + 1),
        mask.shape + (len(Da), len(Dr)),
    ]

    odfs, init_odfs, responses, gaussian_fractions = [math_utils.create_output_array(data, mask, shape) for data, shape in zip((odfs, init_odfs, responses, fs), shapes)]

    reconstructed = sh.calcdwi(sh.sphconv(responses, odfs), grad)
    rmse = np.linalg.norm(reconstructed - dwi, axis=-1) / np.sqrt(dwi.shape[-1])
    rmse *= mask

    # Print execution time
    print(f'Execution time: {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))}')
    return {'odf': odfs, 'response': responses, 'gaussian_fractions': gaussian_fractions, 'rmse': rmse, 'predicted_signal': reconstructed, 'init_odf': init_odfs}


def get_transformation_matrix(num_dirs, lmax):
    """
    Generate a transformation matrix for mapping between diffusion-weighted imaging (DWI) signals and spherical harmonics (SH).

    This function creates a transformation matrix by first generating directions using an external tool (`dirgen`),
    then loading these directions from a text file, and finally computing the transformation matrix using the spherical harmonics basis.

    Parameters:
    - num_dirs (int): The number of directions to generate for the transformation matrix.
    - lmax (int): The maximum order of spherical harmonics to use.
    - cores (int): The number of cores to use for multiprocessing.

    Returns:
    numpy.ndarray: The transformation matrix Q that maps DWI signals to spherical harmonics space.

    Note:
    This function relies on an external command-line tool `dirgen` to generate directions, which are saved to and then read from 'dirs.txt'.
    The file 'dirs.txt' is removed after its contents are loaded.
    """
    subprocess.run(['dirgen', str(num_dirs), os.path.join(os.getcwd(), 'tmp_dirs.txt')])  # Generate directions and save to 'dirs.txt'
    dirs = io_utils.load_vector(os.path.join(os.getcwd(), 'tmp_dirs.txt'))  # Load directions from 'dirs.txt'
    subprocess.run(['rm', 'tmp_dirs.txt'])  # Remove 'dirs.txt' after loading

    # Transformation matrix to map between DWI and SH
    Q = sh.modshbasis(lmax, dirs[:, 0], dirs[:, 1])
    return Q


def decompose_voxel(voxel, Da, Dr, grad, lmax, reg, Q, obj_fun, jac):
    """
    Decompose the voxel into an ODF and a response function.

    Parameters:
    - voxel (numpy.ndarray): The diffusion MRI signal for a single voxel.
    - Da (list or numpy.ndarray): Axial diffusivities for the response function estimation.
    - Dr (list or numpy.ndarray): Radial diffusivities for the response function estimation.
    - grad (numpy.ndarray): The gradient table including b-values and directions. The last column should contain the b-values.
    - lmax (int): Maximum spherical harmonics order.
    - reg (float): Regularization parameter used in the optimization.
    - Q (numpy.ndarray): Transformation matrix mapping DWI signals to spherical harmonics space.
    - obj_fun (function): The objective function for optimization.
    - jac (function): The Jacobian of the objective function.

    Returns:
    dict: A dictionary containing the optimized 'odf', 'response', and 'fs' (Gaussian fractions) for the voxel.

    The optimization process involves several steps:
    1. Simplify b-values processing and constraints setup by rounding and extracting unique b-values, and scaling the signals.
    2. Prepare the initial guess and bounds for the optimization based on the spherical harmonics decomposition.
    3. Execute the optimization step using the `minimize` function from scipy.optimize, applying constraints and regularization.
    4. Prepare the output by extracting the ODF, response function, and Gaussian fractions from the optimization result.
    """
    # Simplify b-values processing and constraints setup
    bvals = np.unique(np.round(grad[..., -1], -2))
    gaussians = sh.zh2rh(get_gaussians(Da, Dr, bvals, lmax))
    S = np.squeeze(sh.calcsig(voxel, np.ones(voxel.shape[:-1], dtype=bool), grad, lmax))
    if S[0,0] == 0:
        return {'odf': np.zeros(sh.n4l(lmax)), 'response': np.zeros((len(bvals), lmax // 2 + 1)), 'gaussian_fractions': np.zeros((len(Da), len(Dr))), 'init_odf': np.zeros(sh.n4l(lmax))}
    scale_factor = 1000 / S[0, 0]
    S *= scale_factor
    gaussians *= S[0, 0]

    # Simplify initial guess and bounds preparation
    init, bounds = get_init_and_bounds_from_csd(lmax, Da, Dr, gaussians, S, [constraints.non_negative_odf(Q)])

    res = minimize(obj_fun, init, jac=jac, bounds=bounds, args= (S, gaussians, reg),
                    constraints=[constraints.non_negative_odf(Q), constraints.sum_of_fractions_equals_one(lmax)],
                    method='SLSQP', options={'ftol':1e-6, 'maxiter': 100000})
    odf = res.x[:sh.n4l(lmax)]
    fs = res.x[sh.n4l(lmax):]

    response = to_response(fs, gaussians) / scale_factor

    return {'odf': odf, 'response': response, 'gaussian_fractions': np.squeeze(fs.reshape((len(Da), len(Dr)))),
            'init_odf': init[:sh.n4l(lmax)], 'init_fs': init[sh.n4l(lmax):].reshape((len(Da), len(Dr)))}

def get_init_and_bounds_from_csd(lmax, Da, Dr, scaled_gaussians, S, constraint_funs):
    """
    Initialise optimisation parameters and bounds for Constrained Spherical Deconvolution (CSD).

    This function prepares the initial guesses for the orientation distribution function (ODF) and the fractions of Gaussian
    basis functions (fs), along with their bounds, for the optimisation process in CSD. It utilizes the normalized L2 weights
    to calculate the initial fs values and sets up bounds based on the spherical harmonics order and the diffusivities.

    Parameters:
    - lmax (int): Maximum spherical harmonics order.
    - Da (numpy.ndarray): Axial diffusivities for the response function estimation.
    - Dr (numpy.ndarray): Radial diffusivities for the response function estimation.
    - scaled_gaussians (numpy.ndarray): Scaled Gaussian distributions representing the response functions.
    - S (numpy.ndarray): The signal from diffusion MRI data for a single voxel.
    - constraint_funs (list): List of constraint functions to be applied during optimisation.

    Returns:
    tuple: A tuple containing two elements:
        - numpy.ndarray: The concatenated initial guesses for the ODF and fs.
        - list: The bounds for the optimisation variables.
    """
    # Initialize bounds for ODF and fs with simplified approach
    bounds = [(1 / np.sqrt(4 * np.pi), 1 / np.sqrt(4 * np.pi)) if i == 0 else (-np.inf, np.inf) for i in range(sh.n4l(lmax))]
    bounds += [(0, 1) if a >= r else (0, 0) for a in Da for r in Dr]

    # Calculate initial fs using normalized L2 weights
    non_zero_fs = np.outer(Da, np.ones(len(Dr))) >= np.outer(np.ones(len(Da)), Dr)
    init_fs = np.ones_like(non_zero_fs, dtype=float)
    # Set initial guess for Gaussian fractions to zero where the condition a >= r is not met
    init_fs[~non_zero_fs] = 0
    init_fs /= init_fs.sum()
    # Flatten the initial guess for Gaussian fractions into a 1D array for optimization
    init_fs = init_fs.flatten()

    # Prepare initial ODF and response function for constrained spherical deconvolution (CSD)
    init_odf = np.zeros(sh.n4l(lmax))
    init_odf[0] = 1 / np.sqrt(4 * np.pi)
    init_rf = np.einsum('a, acd -> cd', init_fs, scaled_gaussians)

    # Perform CSD optimization to refine initial ODF
    temp_csd = minimize(objective_functions.csd_fit, init_odf, (S, init_rf), jac=objective_functions.jac_csd_fit, bounds=bounds[:sh.n4l(lmax)],
                        constraints=constraint_funs, options={'ftol': 1e-2})
    init_odf = temp_csd.x

    return np.concatenate((init_odf, init_fs)), bounds


def get_gaussians(Da, Dr, bvals, lmax):
    """
    Generate zonal Gaussian distributions for given axial and radial diffusivities and b-values.

    Parameters:
    - Da (numpy.ndarray): Axial diffusivities.
    - Dr (numpy.ndarray): Radial diffusivities.
    - bvals (numpy.ndarray): b-values for which the Gaussians are computed.
    - lmax (int): Maximum order of spherical harmonics used in the computation.

    Returns:
    numpy.ndarray: A reshaped array of zonal Gaussian functions for each combination of Da, Dr, and bvals,
                   with dimensions suitable for spherical harmonics of order lmax.
    """
    gaussians = np.zeros((len(Da), len(Dr), len(bvals), lmax // 2 + 1))

    for ir, r in enumerate(Dr):
        for ia, a in enumerate(Da):
            if a >= r:
                gaussians[ia, ir] = sh.zhgaussian(bvals, a, r, lmax)

    return gaussians.reshape((-1, len(bvals), lmax // 2 + 1))


def to_response(fs, gaussians):
    """
    Convert Gaussian fractions to a response function using spherical harmonics.

    This function takes Gaussian fractions (fs) and corresponding Gaussian distributions to compute
    the response function in the spherical harmonics domain. It effectively weights the Gaussian
    distributions by the fractions to obtain a combined response function.

    Parameters:
    - fs (numpy.ndarray): Fractions of Gaussian distributions.
    - gaussians (numpy.ndarray): Gaussian distributions for different diffusivities and b-values.

    Returns:
    numpy.ndarray: The combined response function in the spherical harmonics domain.
    """
    return sh.rh2zh(np.einsum('a, acd -> cd', fs, gaussians))


def expand_response(h):
    """
    Expand a response function to include zero padding for non-zero order terms.

    This function takes a response function defined for zero order terms and expands it to
    a full response function including zero padding for all non-zero order terms. This is mainly
    used to visualise the response functions with MRtrix3. By zero padding we effectively represent
    the response function as an ODF for every shell. By then extracting a shell of interest, we can then
    visualise the response function using sh.load_odf.

    Parameters:
    - h (numpy.ndarray): The input response function of shape (#shells, #orders).

    Returns:
    numpy.ndarray: The expanded response function with zero padding for non-zero order terms,
                   suitable for use with spherical harmonics of order lmax.
    """
    lmax = (h.shape[-1] - 1) * 2
    nlmax = sh.n4l(lmax)
    res = np.zeros(h.shape[:-1] + (nlmax,))
    n2 = 0
    for l in range(0, lmax + 1, 2):
        n1, n2 = n2, (l + 1) * (l + 2) // 2
        res[..., (n1 + n2) // 2] = h[..., l // 2]
    return res

def print_summary(dwi, grad, reg, grid_size, cores):
    """
    Pretty prints the summary of the input parameters.

    Parameters:
    - dwi (numpy.ndarray): The DWI data as a 4D array.
    - grad (numpy.ndarray): The gradient table including b-values and directions.
    - reg (float): Regularisation parameter used in the optimization.
    - grid_size (tuple): The grid size for the optimization.
    - lmax (int): Maximum spherical harmonics order.
    - cores (int): Number of cores to use for multiprocessing.
    """
    max_length = 50  # Adjust this value as needed to ensure proper alignment
    print("#"*(max_length+1))
    print("# Summary:".ljust(max_length) + "#")
    print('#' + '-'*(max_length-1) + '#')
    print(f"# Size of DWI input: {dwi.shape}".ljust(max_length) + "#")
    print(f"# Regularisation parameter: {reg:.1e}".ljust(max_length) + "#")
    print(f"# Gaussian grid size: {grid_size} x {grid_size}".ljust(max_length) + "#")
    print(f"# Number of cores: {cores}".ljust(max_length) + "#")
    gradient_utils.pretty_print_gradient_table(grad, max_length)
    print("#"*(max_length+1))
    
