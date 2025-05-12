import numpy as np

def rmse(dwi1, dwi2, mask):
    """
    Root Mean Squared Error of 2 DWI images. RMSE is calculated on the voxel level.
    The number of samples thus equals the number of gradient directions.
    :param dwi1: First DWI signal
    :param dwi2: Second DWI signal, shape must match dwi1.
    :param mask: Binary brain mask. RMSE is only calculated for brain voxels.
    :return: Array with per-voxel RMSE values.
    """
    assert dwi1.shape == dwi2.shape, f'Input shapes do not align. dwi1 has shape {dwi1.shape} while dwi2 has shape {dwi2.shape}'
    res = np.zeros(dwi1.shape[:-1])
    res[mask] = np.linalg.norm(dwi1[mask] - dwi2[mask], ord=2, axis=-1) / np.sqrt(dwi1.shape[-1])
    return res

def create_output_array(data, mask, output_shape):
    """
    Create an output array from flat input data. Initialise an empty array with shape output_shape.
    `data` is then written to those locations where the brain mask is non-zero.
    :param data: Input data with as many elements as there are True values in `mask`.
    :param mask: Binary brain mask.
    :param output_shape: Output shape of the resulting data.
    :return: Array with shape `output_shape` with voxels within the brain mask set to `data`.
    """
    output_arr = np.zeros(output_shape, dtype=data.dtype)
    output_arr[mask] = data
    return output_arr


def normalized_rmse(list1, list2):

    # Calculate RMSE
    rmse = np.sqrt(np.mean((list1 - list2) ** 2, axis=-1))

    # Calculate Mean Normalized RMSE
    mean_normalized_rmse = rmse / np.mean(list1, axis=-1)

    return mean_normalized_rmse
