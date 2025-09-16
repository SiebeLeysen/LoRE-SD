"""Plot a learned contrast matrix as a square heatmap with colorbar.

Usage:
    python -m lore_sd.plot_contrast --C C.txt --out C.png --vmin 0 --vmax 1

Saves a PNG (or shows image if --out is omitted).
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_contrast(C_path: str, out_path: str | None = None, cmap: str = 'inferno', vmin: float | None = None, vmax: float | None = None):
    C = np.loadtxt(C_path)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        # still allow rectangular but warn
        pass

    fig, ax = plt.subplots(figsize=(6, 6))
    # show with origin='upper' so the largest y-value (top of matrix) appears at the bottom
    im = ax.imshow(C, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)
    # Axial (y) and radial (x) diffusivity labels spanning 0..4 ms/μm^2
    ax.set_ylabel(r'$\lambda_\parallel (ms/μm^2)$')
    ax.set_xlabel(r'$\lambda_\perp (ms/μm^2)$')
    # smaller colorbar for balanced figure
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Weight')
    ax.set_aspect('equal')
    # set ticks to map pixel indices to physical range 0..4
    A, B = C.shape
    nticks = 5
    xticks = np.linspace(0, A - 1, nticks)
    yticks = np.linspace(0, B - 1, nticks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(0, 4, nticks)])
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 4, nticks)])
    fig.tight_layout()
    if out_path is None:
        plt.show()
    else:
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def _build_parser():
    p = argparse.ArgumentParser(description='Plot contrast matrix heatmap')
    p.add_argument('--C', '-c', required=True, help='Contrast text file (A x B)')
    p.add_argument('--out', '-o', default=None, help='Output image file (png, pdf, svg). If omitted shows interactively.')
    p.add_argument('--cmap', default='viridis', help='Matplotlib colormap')
    p.add_argument('--vmin', type=float, default=None, help='Colorbar minimum')
    p.add_argument('--vmax', type=float, default=None, help='Colorbar maximum')
    return p


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    p = _build_parser()
    args = p.parse_args(argv)
    plot_contrast(args.C, args.out, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax)


if __name__ == '__main__':
    main()
