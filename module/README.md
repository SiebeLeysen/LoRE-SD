# LoRE-SD MRtrix External Module

LoRE-SD is a diffusion MRI spherical deconvolution method that jointly estimates:

- a local spherical harmonic orientation distribution function (ODF),
- a grid of Gaussian fraction weights over axial and radial diffusivities, and
- a local per-voxel response function.

The optimization is formulated as an alternating least-squares (ALS) procedure that alternates between:

1. ODF estimation under non-negativity constraints.
2. Gaussian fraction estimation under non-negativity constraints.

Both subproblems are solved using ADMM (Alternating Direction Method of Multipliers), removing the need for external nonlinear optimization libraries.

This module packages LoRE-SD as an MRtrix external module with two commands:

- `lore_dwi2fod`: fit ODFs, fractions, and response functions from diffusion MRI data.
- `lore_fractions2contrasts`: convert fitted fractions into biologically motivated contrast maps.

## Repository layout

- `cmd/lore_dwi2fod.cpp` — LoRE-SD fitting command
- `cmd/lore_fractions2contrasts.cpp` — contrast mapping command
- `src/lore_sd/lore_sd.cpp` — core model fitting and optimization
- `src/lore_sd/lore_sd.h` — public API

## Build

This module uses the standard MRtrix external module build system.

No additional third-party optimization libraries are required.

From the repository root, create a symbolic link to the MRtrix build script if one is not already present, then build the module:

```bash
ln -s /path/to/mrtrix3/build build
./build
```

## Method overview

For each voxel:

1. Shell-wise spherical harmonic representations of the diffusion signal are estimated.
2. An initial isotropic Gaussian fraction distribution is constructed.
3. An initial isotropic ODF is constructed.
4. The algorithm alternates between:
   - ODF estimation using constrained ADMM,
   - Gaussian fraction estimation using non-negative ADMM,
   - response function reconstruction.
5. Iteration continues until the objective function stabilizes or the maximum number of ALS iterations (20) is reached.

The ODF non-negativity constraints are enforced using a dense spherical sampling of directions obtained from the predefined MRtrix electrostatic repulsion set.

## Usage

### 1) Fit LoRE-SD from DWI

Basic example:

```bash
lore_dwi2fod dwi.mif odf.mif fracs.mif response.mif \
  -mask mask.mif \
  -lmax 8 \
  -grid_size 10 \
  -reg 1e-3 \
  -python_shells \
  -init_obj_fun init_obj.mif \
  -final_obj_fun final_obj.mif
```

### Inputs

- diffusion-weighted image (DWI) in MRtrix format
- optional mask via `-mask`
- optional shell grouping via `-python_shells`

### Outputs

- `odf.mif`: spherical harmonic ODF coefficients
- `fracs.mif`: Gaussian fraction grid over diffusivity space
- `response.mif`: fitted response function coefficients
- `init_obj.mif`: initial objective value per voxel (optional)
- `final_obj.mif`: final objective value per voxel (optional)

### Main options

- `-lmax <int>`: maximum spherical harmonic order (default: 8)
- `-grid_size <int>`: axial/radial diffusivity grid size (default: 10)
- `-reg <float>`: response-function regularization strength (default: `1e-3`)
- `-init_obj_fun <image>`: write initial objective values
- `-final_obj_fun <image>`: write final objective values

The ODF non-negativity constraints use the built-in MRtrix predefined 129-direction electrostatic repulsion set.

### Optimization diagnostics

Optional objective images can be exported for quality control:

- `init_obj_fun`: objective value before optimization
- `final_obj_fun`: objective value after optimization

These outputs can be useful for identifying convergence failures or problematic voxels.

## 2) Generate contrasts from fractions

The contrast command converts the fitted diffusivity fractions into scalar contrast maps:

- intra-axonal contrast
- extra-axonal contrast
- free-water contrast
- relative fractional anisotropy (RFA)

Example:

```bash
lore_fractions2contrasts fracs.mif \
  -intra_axonal intra.mif \
  -extra_axonal extra.mif \
  -free_water free.mif \
  -rfa rfa.mif
```

Useful options:

- `-rate <int>`: decay rate for intra-axonal weighting (default: 10)
- `-with_isotropic`: include the isotropic line (`AD = RD`) in the valid region

## Contrast generation details

The fraction image is interpreted as a discretized diffusivity plane parameterized by:

- axial diffusivity (AD)
- radial diffusivity (RD)

The following maps are then generated:

- **free-water**: high-diffusivity isotropic regime
- **intra-axonal**: highly anisotropic regime
- **extra-axonal**: remaining non-free-water tissue regime
- **RFA**: anisotropy-derived scalar computed from diffusivity differences

## Implementation notes

- The ODF DC coefficient is fixed to the isotropic value `1/sqrt(4π)`.
- Shell-wise spherical harmonic fits are used as the signal representation.
- Gaussian basis functions are defined on a discretized `(Da, Dr)` grid.
- The Gaussian basis ordering is Da-major followed by Dr-major.
- ODF estimation is performed using constrained ADMM.
- Fraction estimation is performed using non-negative ADMM.
- Response functions are reconstructed directly from the fitted fraction distribution.
- Objective values can optionally be exported for quality control.
- Non-negativity constraints use the MRtrix predefined electrostatic repulsion directions.

## Example workflow

1. Build the module.
2. Run `lore_dwi2fod` on a DWI dataset using a brain mask.
3. Inspect:
   - `odf.mif`
   - `fracs.mif`
   - `response.mif`
   - optional objective maps.
4. Generate contrast maps using `lore_fractions2contrasts`.


## Author

Siebe Leysen
