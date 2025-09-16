"""Learn a non-negative contrast matrix from a representation and target contrast.

Usage (CLI):
    python -m lore_sd.learn_contrast --rep rep.mif --contrast contrast.mif --out C.txt

The representation file should be an MRtrix .mif with shape (X,Y,Z,A,B).
The contrast file should be an MRtrix .mif with shape (X,Y,Z).
The output is a whitespace-separated text file containing the learned matrix C of shape (A,B).

The solver uses scipy.optimize.lsq_linear (preferred) or scipy.optimize.nnls when available.
If scipy is not available a simple multiplicative update fallback is used (slower / approximate).
"""
from __future__ import annotations

import argparse
import sys
import numpy as np
from typing import Tuple

from lore_sd.mrtrix_io.io.image import load_mrtrix


def _prepare_matrix(rep: np.ndarray, contrast: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Build design matrix M (voxels x (A*B)) and target vector y (voxels,).

    Filters out voxels where contrast is NaN or where representation is all zeros. Optionally apply mask.
    Returns M, y and shape (A,B).
    """
    # representation should have exactly two extra dimensions compared to contrast
    if rep.ndim != contrast.ndim + 2:
        raise ValueError(f"Representation must have two extra dims compared to contrast. Got rep{rep.shape}, contrast{contrast.shape}")

    spatial_shape = rep.shape[:-2]
    A, B = rep.shape[-2], rep.shape[-1]

    if contrast.shape != spatial_shape:
        raise ValueError("Contrast and representation spatial dimensions do not match.")

    # Default mask: all True with same spatial shape
    if mask is None:
        mask = np.ones(spatial_shape, dtype=bool)
    else:
        if mask.shape != spatial_shape:
            raise ValueError(f"Mask must have shape {spatial_shape}.")

    # Select voxels by mask and build design matrix
    # rep[mask] has shape (n_valid, A, B)
    rep_sel = rep[mask]
    y_sel = contrast[mask]

    if rep_sel.size == 0:
        raise ValueError("No valid voxels found to fit.")

    M = rep_sel.reshape((rep_sel.shape[0], A * B)).astype(float)
    y = y_sel.astype(float).reshape((rep_sel.shape[0],))
    return M, y, (A, B)


def solve_nnls_scipy(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve non-negative least squares using scipy if available."""
    try:
        from scipy.optimize import lsq_linear

        # bounds (0, inf) enforces non-negativity
        res = lsq_linear(M, y, bounds=(0, np.inf), lsmr_tol='auto', verbose=0)
        if not res.success:
            # fall back to nnls if lsq_linear fails
            from scipy.optimize import nnls
            c, rnorm = nnls(M, y)
            return c
        return res.x
    except Exception:
        # try nnls as fallback
        try:
            from scipy.optimize import nnls

            c, rnorm = nnls(M, y)
            return c
        except Exception:
            raise


def learn_contrast(rep_paths: str, contrast_paths: str, out_path: str, mask_paths: str | None = None) -> np.ndarray:
    """Main entry: load one or more images, concatenate data and solve NNLS.

    rep_paths / contrast_paths may be comma-separated lists. If a single contrast path is
    provided and multiple reps are given, the same contrast is reused for all reps. Masks
    (optional) can also be provided as a comma-separated list; use an empty entry to skip a mask.

    Returns learned C as ndarray shape (A,B).
    """
    rep_list = rep_paths.split(',') if isinstance(rep_paths, str) else list(rep_paths)
    contrast_list = contrast_paths.split(',') if isinstance(contrast_paths, str) else list(contrast_paths)
    if len(contrast_list) == 1 and len(rep_list) > 1:
        # reuse same contrast for all reps
        contrast_list = [contrast_list[0]] * len(rep_list)
    if len(contrast_list) != len(rep_list):
        raise ValueError('Number of contrast files must be 1 or equal to number of representation files')

    mask_list = None
    if mask_paths is not None:
        mask_list = mask_paths.split(',') if isinstance(mask_paths, str) else list(mask_paths)
        if len(mask_list) == 1 and len(rep_list) > 1:
            mask_list = [mask_list[0]] * len(rep_list)
        if len(mask_list) != len(rep_list):
            raise ValueError('Number of mask files must be 1 or equal to number of representation files')

    M_list = []
    y_list = []
    A_B = None
    for idx, rpath in enumerate(rep_list):
        rep_img = load_mrtrix(rpath)
        contrast_img = load_mrtrix(contrast_list[idx])
        mask_img = None
        if mask_list is not None and mask_list[idx] not in (None, '', 'None'):
            mask_img = load_mrtrix(mask_list[idx])

        M_i, y_i, (A, B) = _prepare_matrix(rep_img.data, contrast_img.data, None if mask_img is None else mask_img.data > 0.5)
        if A_B is None:
            A_B = (A, B)
        else:
            if A_B != (A, B):
                raise ValueError(f'Representation shapes (A,B) must match across inputs; got {A_B} and {(A,B)}')
        M_list.append(M_i)
        y_list.append(y_i)

    M = np.vstack(M_list)
    y = np.concatenate(y_list)

    # Use SciPy-based NNLS solver (lsq_linear or nnls). Let exceptions propagate.
    c = solve_nnls_scipy(M, y)

    C = c.reshape(A_B)

    # Columns in M that are all zero correspond to (a,b) pairs that were not allowed to have values.
    # Set those entries in C to NaN to mark them as invalid.
    col_sums = np.sum(np.abs(M), axis=0)
    zero_cols = (col_sums == 0)
    if np.any(zero_cols):
        C_flat = C.reshape((-1,))
        C_flat[zero_cols] = np.nan
        C = C_flat.reshape(A_B)

    # Save as text file
    np.savetxt(out_path, C, fmt='%.12g')
    return C


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Learn non-negative contrast matrix from representation')
    p.add_argument('--rep', '-r', required=True, help='Representation .mif file (X,Y,Z,A,B)')
    p.add_argument('--contrast', '-c', required=True, help='Target contrast .mif file (X,Y,Z)')
    p.add_argument('--out', '-o', required=True, help='Output text file for C (A x B)')
    p.add_argument('--mask', '-m', default=None, help='Optional mask .mif file (X,Y,Z). Only voxels with non-zero mask used.')
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = _build_parser()
    args = p.parse_args(argv)

    C = learn_contrast(args.rep, args.contrast, args.out, args.mask)
    print(f"Saved learned contrast matrix to {args.out} with shape {C.shape}")


if __name__ == '__main__':
    main()
