# LoRE-SD MRtrix external module

LoRE-SD is a diffusion MRI spherical deconvolution method that estimates:

- a local spherical harmonic ODF,
- a grid of Gaussian fraction weights over axial/radial diffusivities, and
- a local, per-voxel response function.

This module packages LoRE-SD as an MRtrix external module with two commands:

- lore_dwi2fod: fit ODFs, fractions, and response functions from DWI
- lore_fractions2contrasts: map the fitted fractions to biologically motivated contrasts

## Repository layout

- [cmd/lore_dwi2fod.cpp](cmd/lore_dwi2fod.cpp) — LoRE-SD fitting command
- [cmd/lore_fractions2contrasts.cpp](cmd/lore_fractions2contrasts.cpp) — contrast mapping command
- [src/lore_sd/lore_sd.cpp](src/lore_sd/lore_sd.cpp) — core fitting and model construction
- [src/lore_sd/lore_sd.h](src/lore_sd/lore_sd.h) — public API for the fitter
- MRtrix predefined directions — built-in 300-direction electrostatic repulsion set used for non-negativity constraints

## Build

This module is built using the MRtrix build script.

This module depends on the nlopt library for nonlinear optimization.
To build this module with NLopt, the NLopt library must be available and linked in the MRtrix configuration file, typically named `config` in the MRtrix repository.
Add the following lines: 

```
cpp_flags += [
    '-I/path/to/nlopt/include'
]

ld_flags += [
    '-L/path/to/nlopt/lib',
    '-lnlopt',
    '-Wl,-rpath,/path/to/nlopt/lib'
]
```


From the repository root, create a symbolic link to the MRtrix build script if one is not already present, then run it:

```bash
ln -s /path/to/mrtrix3/build build
./build
```

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

- DWI image in MRtrix format
- optional mask via -mask
- optional shell grouping via -python_shells

### Outputs

- odf.mif: spherical harmonic ODF coefficients
- fracs.mif: Gaussian fraction grid with shape (ad, rd)
- response.mif: per-shell response function in SH form
- init_obj.mif: initial objective value per voxel
- final_obj.mif: final objective value per voxel

### Main options

- -lmax <int>: spherical harmonic order, default 8
- -grid_size <int>: Da/Dr grid size, default 10
- -reg <float>: regularisation strength, default 1e-3
- -maxeval <int>: maximum iterations for the main optimizer, default 400
- -python_shells: group shells by rounding b-values to the nearest 100
- -init_obj_fun <image>: write initial objective values per voxel
- -final_obj_fun <image>: write final objective values per voxel

The non-negativity constraint directions use the built-in MRtrix predefined 300-direction electrostatic repulsion set.
If you want a different set of directions, change the predefined set selected in the command wrapper.

### 2) Generate contrasts from fractions

The contrast command converts the 5D fractions image into scalar maps that summarize different tissue regimes:

- intra-axonal contrast
- extra-axonal contrast
- free-water contrast
- RFA map

Example:

```bash
lore_fractions2contrasts fracs.mif \
  -intra_axonal intra.mif \
  -extra_axonal extra.mif \
  -free_water free.mif \
  -rfa rfa.mif
```

Useful options:

- -rate <int>: decay rate used for the intra-axonal weighting, default 10
- -with_isotropic: include the isotropic line ad == rd in the valid region

## Contrast generation details

The contrast mapping command interprets the fractions image as a grid over axial diffusivity (ad) and radial diffusivity (rd). It then computes:

- free-water: a binary mask over high-diffusivity combinations,
- intra-axonal: a decaying weighting over the valid ad >= rd region,
- extra-axonal: the complement of free-water and intra-axonal contributions,
- RFA: an anisotropy-style scalar derived from ad/rd.

## Implementation notes

- The ODF DC term is fixed to the isotropic value.
- Shell-wise SH fits are used to build the initial response estimate.
- Gaussian basis ordering is Da-major, then Dr-major, matching the Python implementation.
- Objective values can be exported per voxel for QC.
- Non-negativity directions are taken from the MRtrix predefined directions API.

## Example workflow

1. Build the module.
2. Run lore_dwi2fod with a brain mask.
3. Inspect odf.mif, fracs.mif, response.mif, and optional objective images.
4. Run lore_fractions2contrasts to generate contrast maps from the fractions volume.

## License

A license file is not included in this repository snapshot.

## Author

Siebe Leysen
