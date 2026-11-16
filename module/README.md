# LoRE-SD MRtrix External Module

This MRtrix3 external module provides a multidimensional LoRE-SD model. It
jointly estimates a spherical harmonic orientation distribution function
(ODF), a local response function, and non-negative weights over a grid of
axial diffusivity (AD), radial diffusivity (RD), and, when supported by the
acquisition, T2 values.

The fitting command alternates between constrained ODF estimation and
non-negative fraction estimation, using ADMM for both subproblems.

## Commands and source

- `lore_dwi2fod`: fit an ODF, fractions, and response function from a DWI.
- `lore_fractions2contrasts`: map fitted fractions to scalar contrast images.
- `cmd/lore_dwi2fod.cpp` and `src/lore_sd/lore_sd_multidim.cpp`: fitting command
  and multidimensional model implementation.
- `cmd/lore_fractions2contrasts.cpp`: contrast mapping, including reduction
  across the optional T2 axis.

## Build

This module uses the standard MRtrix external-module build system. From the
`module` directory, link the MRtrix build script if needed and build:

```bash
ln -s /path/to/mrtrix3/build build
./build
```

## Fit LoRE-SD from a DWI

```bash
lore_dwi2fod dwi.mif odf.mif fracs.mif response.mif \
  -mask mask.mif \
  -lmax 8 \
  -grid_da 10 \
  -grid_dr 10 \
  -grid_t2 10 \
  -reg 4e-5
```

The input image must have a diffusion gradient scheme in its header, or receive
one using `-grad` or `-fslgrad`.

### Outputs

- `odf.mif`: spherical harmonic ODF coefficients.
- `fracs.mif`: fraction weights on the AD x RD x T2 grid. This is a 6D image
  with dimensions `(x,y,z,AD,RD,T2)`; the T2 axis has length one when T2 fitting
  is disabled. Values are ordered AD-major, then RD, then T2.
- `response.mif`: response coefficients by acquisition shell and even SH
  order.
- Optional `-init_obj_fun <image>` and `-final_obj_fun <image>`: initial and
  final objective values per voxel.
- Optional `-predicted_signal <image>`: fitted signal on the input acquisition
  scheme.
- Optional `-recon_grad_mrtrix <file> -dwi_recon <image>`: predicted signal on
  a separate acquisition scheme.

### Main options

- `-mask <image>`: process only voxels in this binary mask.
- `-lmax <int>`: maximum even spherical harmonic order (default: `8`).
- `-grid_da <int>`: number of AD grid points (default: `10`, minimum: `2`).
- `-grid_dr <int>`: number of RD grid points (default: `10`, minimum: `1`).
- `-grid_t2 <int>`: number of T2 grid points (default: `10` when multiple echo
  times are present; minimum: `1`).
- `-reg <float>`: response-function regularization strength (default: `4e-5`).

AD and RD are linearly sampled from 0 to `3.3e-3` mm²/s. T2 fitting is
automatically disabled if no TE is supplied or all measurements have the same
TE, in which case the T2 axis has length one regardless of `-grid_t2`. For a
For T2 grid sizes of two or more, the grid starts at 30 ms and includes a 1500
ms endpoint. For three or more points, intermediate grid points span 30–170 ms.

### Gradient and acquisition options

- `-grad <file>`: gradient table in MRtrix/scanner coordinates, one row per
  volume: `X Y Z b`, optionally followed by `bdelta` and `TE` columns (5- or
  6-column table).
- `-fslgrad <bvecs> <bvals>`: use FSL-format gradient files instead of `-grad`.
- `-bdelta <file>` and `-te <file>`: provide b-tensor shape and echo time
  values. Each file may contain one value per b-shell or per volume. They
  cannot be used with a `-grad` table that already contains the corresponding
  column. TE values are in milliseconds.
- `-recon_grad_mrtrix <file> -dwi_recon <image>`: predict on another MRtrix
  gradient table, with the same 4-, 5-, or 6-column format as `-grad`. The
  reconstruction table's optional protocol columns default to `bdelta=1` and
  `TE=0`; if the gradient scheme matches the input, omitted protocol values
  inherit the input values.

The command groups measurements into shells using b-value, bdelta, and TE.
When no explicit gradients are provided, the scheme stored in the input image
header is used.

## Generate contrasts from fractions

```bash
lore_fractions2contrasts fracs.mif \
  -intra_axonal intra.mif \
  -extra_axonal extra.mif \
  -free_water free.mif \
  -rfa rfa.mif \
  -ad ad.mif \
  -rd rd.mif \
  -tc_rfa tc_rfa.mif
```

Select one or more output maps by supplying their options. The command accepts
legacy 5D `(x,y,z,AD,RD)` fractions and 6D `(x,y,z,AD,RD,T2)` fractions. For
6D input it sums fractions over T2 before calculating each contrast, so all T2
components contribute to the corresponding AD/RD map.

Optional controls:

- `-rate <int>`: decay rate for intra-axonal weighting (default: `10`).
- `-with_isotropic`: include the isotropic line (`AD = RD`) in the valid region.

The generated maps include intra-axonal, extra-axonal, and free-water
fractions, relative fractional anisotropy (RFA), AD, RD, and tissue-conditioned
RFA (`-tc_rfa`).

## Method notes

- Shell-wise spherical harmonic signal representations are used.
- Initial fractions include valid anisotropic atoms; the initial response is
  scaled using b=0 measurements when available.
- The ODF DC coefficient is fixed to `1/sqrt(4π)`.
- ODF non-negativity is enforced using MRtrix's predefined 129-direction
  electrostatic repulsion set.
- The AD/RD/T2 basis is stored AD-major, then RD, then T2; atoms with `AD < RD`
  are excluded.
- The fitting loop performs up to 20 alternating optimization steps.

## Example workflow

1. Build the module.
2. Run `lore_dwi2fod` with a gradient scheme and mask.
3. Inspect the ODF, fraction, and response images (and optional diagnostics).
4. Run `lore_fractions2contrasts` on the 6D fraction image.

## Author

Siebe Leysen
