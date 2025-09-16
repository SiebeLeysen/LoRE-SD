"""Apply a learned contrast matrix C to a representation .mif to produce a contrast image (.mif).

Usage (CLI):
    python -m lore_sd.apply_contrast --rep rep.mif --C C.txt --out contrast.mif

The representation file should be an MRtrix .mif with shape (X,Y,Z,A,B).
The contrast file is a whitespace-separated text file with shape (A,B).
The output image will be a .mif containing the computed contrast per voxel with shape (X,Y,Z) and the same header metadata as the input rep.
"""
from __future__ import annotations

import argparse
import sys
import numpy as np
from lore_sd.mrtrix_io.io.image import load_mrtrix, save_mrtrix, Image


def apply_contrast(rep_path: str, C_path: str, out_path: str, mask_path: str | None = None) -> Image:
    """Load representation, load C, compute contrast map and save as .mif Image.

    Returns the Image object saved.
    """
    rep_img = load_mrtrix(rep_path)
    rep = rep_img.data
    if rep.ndim < 2:
        raise ValueError(f"Representation must have at least 2 dims. Got {rep.shape}")
    spatial_shape = rep.shape[:-2]
    A, B = rep.shape[-2], rep.shape[-1]

    C = np.nan_to_num(np.loadtxt(C_path))  # convert NaNs to zero to avoid issues in einsum
    if C.shape != (A, B):
        raise ValueError(f"Contrast matrix shape mismatch: file {C_path} has shape {C.shape}, expected {(A,B)}")

    mask = None
    if mask_path is not None:
        mask_img = load_mrtrix(mask_path)
        mask = mask_img.data.astype(bool)
        if mask.shape != spatial_shape:
            raise ValueError(f"Mask must have shape {spatial_shape}")

    # vectorized computation: sum over last two dims a,b
    out = np.einsum('...ab,ab->...', rep, C)
    if mask is not None:
        out = np.where(mask, out, 0.0)

    out_img = Image(data=out, vox=rep_img.vox, transform=rep_img.transform, grad=None, comments='')
    save_mrtrix(out_path, out_img)
    return out_img


def _build_parser():
    p = argparse.ArgumentParser(description='Apply contrast matrix to representation .mif')
    p.add_argument('--rep', '-r', required=True, help='Representation .mif file (X,Y,Z,A,B)')
    p.add_argument('--C', '-c', required=True, help='Contrast text file (A x B)')
    p.add_argument('--out', '-o', required=True, help='Output .mif file for contrast (X x Y x Z)')
    p.add_argument('--mask', '-m', default=None, help='Optional mask .mif to limit computation')
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = _build_parser()
    args = p.parse_args(argv)
    apply_contrast(args.rep, args.C, args.out, args.mask)
    print(f"Saved contrast image to {args.out}")


if __name__ == '__main__':
    main()
