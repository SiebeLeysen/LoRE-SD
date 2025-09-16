import argparse
import numpy as np
import os
import sys
import subprocess
import numba

# sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD/src')
# sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD')

from lore_sd.optimisation import contrasts
from lore_sd.optimisation import optimise
from lore_sd.utils import SphericalHarmonics as sh

from lore_sd.mrtrix_io.io import load_mrtrix, save_mrtrix
from lore_sd.mrtrix_io.io.image import Image

def handle_input(args):
    """
    Handles the input image file based on its format.

    If the input image is not in '.mif' format, it converts the image to '.mif' using the provided bvecs and bvals files.
    Otherwise, it directly loads the '.mif' file.

    Parameters:
    - args (Namespace): Command line arguments containing the input image path, bvecs, and bvals file paths.

    Returns:
    - Image: The loaded or converted MRtrix image object.

    Raises:
    - AssertionError: If the input image is not in '.mif' format and either bvecs or bvals files are not provided.
    """
    if not args.input.endswith('.mif'):
        raise Exception('Input file should be in MRtrix3 format (.mif)')
    else:
        return load_mrtrix(args.input)

def adjust_cores(requested_cores):
    """
    Adjusts the number of cores to use based on the system's capabilities and the user's request.

    Parameters:
    - requested_cores (int): The number of cores requested by the user.

    Returns:
    - int: The actual number of cores to use, which is the minimum of the requested cores and the system's available cores.
    """
    max_cores = os.cpu_count()
    return min(requested_cores, max_cores)

def prepare_parameters(grid_size):
    """
    Prepares the parameters for the signal decomposition process.

    Generates linearly spaced values for Axial Diffusivity (AD) and Radial Diffusivity (RD) based on the specified grid size.

    Parameters:
    - grid_size (int): The number of linearly spaced values to generate for AD and RD.

    Returns:
    - tuple: A tuple containing two numpy arrays for AD and RD values.
    """
    ad_list = np.linspace(0, 4e-3, grid_size)
    rd_list = np.linspace(0, 4e-3, grid_size)
    return ad_list, rd_list

def save_outputs(args, data_dict, vox, grad):
    os.makedirs(args.output_dir, exist_ok=True)
    for file_name, data in data_dict.items():
        if file_name == 'predicted_signal.mif':
            save_mrtrix(os.path.join(args.output_dir, file_name), Image(data, grad=grad, vox=vox, comments=f'{file_name.split(".")[0].replace("_", " ").title()} by LoRE-SD'))
        else:
            save_mrtrix(os.path.join(args.output_dir, file_name), Image(data, vox=vox, comments=f'{file_name.split(".")[0].replace("_", " ").title()} by LoRE-SD'))

def get_mask(input_image, cores):
    subprocess.run(['dwi2mask', input_image, 'tmp_mask.mif', '-nthreads', str(cores)])
    mask = load_mrtrix('tmp_mask.mif').data > .5
    subprocess.run(['rm', 'tmp_mask.mif'])
    return mask

def main():
    parser = argparse.ArgumentParser(
        description='Decompose the dMRI data into voxel-level ODFs and response functions using LoRE-SD.')
    parser.add_argument('input', help='Input image')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--reg', help='Regularisation parameter', type=float, default=1e-3)
    parser.add_argument('--grid_size', help='Grid size (Number of linearly spaced values for AD and RD)', type=int,
                        default=10)
    parser.add_argument('--cores', help='Number of cores to use', type=int, default=1)
    parser.add_argument('--mask', help='Path to the mask file', default=None)
    parser.add_argument('--eval_matrix', help='Path to the evaluation matrix', default=None)
    parser.add_argument('--slice', help='Slice number to process', type=int, default=None)

    args = parser.parse_args()

    input_args = handle_input(args)
    cores = adjust_cores(args.cores)
    if args.mask is not None:
        mask = load_mrtrix(args.mask).data > .5
    else:
        mask = get_mask(args.input, cores)
    
    ad_list, rd_list = prepare_parameters(args.grid_size)

    if args.eval_matrix is not None:
        # load the data in np format
        Q = np.load(args.eval_matrix)
    else:
        Q = optimise.get_transformation_matrix(300, 8)

    dwi = input_args.data

    # Single slice processing
    if args.slice is not None:
        dwi = dwi[:,:,args.slice:args.slice+1]
        mask = mask[:,:,args.slice:args.slice+1]

    grad = input_args.grad
    grad[:, -1] = np.round(grad[:, -1], -2)

    optimise.print_summary(dwi, grad, args.reg, args.grid_size, cores)

    out = optimise.get_signal_decomposition(dwi, mask, grad, ad_list, rd_list, args.reg, Q=Q, cores=cores)


    vox = load_mrtrix(args.input).vox

    save_dict = {
        'odf.mif': out['odf'],
        'response.mif': out['response'],
        'gaussian_fractions.mif': out['gaussian_fractions'],
        'predicted_signal.mif': out['predicted_signal'],
        'rmse.mif': out['rmse'],
        'init_odf.mif': out['init_odf']
    }

    save_outputs(args, save_dict, vox, grad)


if __name__ == '__main__':
    main()
