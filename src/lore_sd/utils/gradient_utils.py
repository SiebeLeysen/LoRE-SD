import numpy as np
import os
import subprocess
from .io_utils import load_vector

def pretty_print_gradient_table(grad, max_length=None):
    """
    Print the gradient table as a list of b-values and number of directions for each b-value.
    :param grad: Gradient table
    :return: None
    """
    b_values = grad[:, -1]
    unique_b_values, counts = np.unique(np.round(b_values, -2), return_counts=True)
    b_value_counts = dict(zip(unique_b_values, counts))
    
    if max_length is None:
        print('# b-value   |   count   ')
        print('# ----------------------')
    else:
        print('# b-value   |   count   '.ljust(max_length) + "#")
        print('#' + '-'*(max_length-1) + '#')

    for b in unique_b_values:
        if max_length is None:
            print(f"# {int(b):5d}     |   {b_value_counts[b]}" + "#")
        else:
            print(f"# {int(b):5d}     |   {b_value_counts[b]}".ljust(max_length) + "#")


def create_gradient_table(bvals, dirs_per_shell):
    """
    Create a gradient table with a specific number of directions per shell.

    Args:
        bvals (list): List of b-values.
        dirs_per_shell (list): List of number of directions per shell.

    Returns:
        numpy.ndarray: A gradient table with the specified number of directions per shell.
    """
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    grad = np.empty((np.sum(dirs_per_shell), 4))
    start = 0

    for bval, dirs in zip(bvals, dirs_per_shell):
        subprocess.run(['dirgen', str(dirs), os.path.join(tmp_dir, 'tmp_dirs.txt'), '-cart'])
        dirs = load_vector(os.path.join(tmp_dir, 'tmp_dirs.txt'))
        subprocess.run(['rm', os.path.join(tmp_dir, 'tmp_dirs.txt')])
        grad[start:start+dirs.shape[0], :-1] = dirs
        grad[start:start+dirs.shape[0], -1] = bval

        start += dirs.shape[0]

    subprocess.run(['rmdir', tmp_dir])

    return grad