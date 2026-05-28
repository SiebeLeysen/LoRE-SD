# LoRE-SD MRtrix external module

LoRE-SD is a diffusion MRI fibre response decomposition method that estimates:

- a spherical harmonic ODF,
- a grid of Gaussian fraction weights, and
- a per-shell response function.

This repository packages the implementation as an MRtrix external module with a single command:

- `dwi2fod_lore_sd`

## Repository layout

- `cmd/dwi2fod_lore_sd.cpp` — MRtrix command-line wrapper
- `src/lore_sd/lore_sd.cpp` — core fitting and model construction
- `src/lore_sd/lore_sd.h` — public API for the fitter

## Build

This module is built using the MRtrix build script.

1. Make sure NLopt is installed on your system.
2. Add NLopt to the MRtrix build configuration used by this module:
  - `cpp_flags` should include the NLopt include flags
  - `ld_flags` should include the NLopt link flags as well

From the repository root, create a symbolic link to the MRtrix `build` script if one is not already present, then run it:

```bash
ln -s /path/to/mrtrix3/build build
./build
```

This builds the `dwi2fod_lore_sd` command using the MRtrix build system.

## Usage

Basic example:

```bash
dwi2fod_lore_sd dwi.mif odf.mif fracs.mif response.mif \
  -nthreads N
```

### Inputs

- `dwi.mif`: diffusion-weighted input image
- optional mask image via `-mask`
- optional non-negativity directions via `-directions`

### Outputs

- `odf.mif`: output ODF coefficients
- `fracs.mif`: Gaussian fraction grid
- `response.mif`: per-shell response function
- optional `--init_obj_fun <image>`: objective value before optimization (QC)
- optional `--final_obj_fun <image>`: objective value after optimization (QC)

## Main options

- `-lmax <int>`: spherical harmonic order, default `8`
- `-grid_size <int>`: Da/Dr grid size, default `7`
- `-reg <float>`: regularisation strength, default `1e-3`
- `-maxeval <int>`: maximum iterations for the main optimizer, default `400`
- `-python_shells`: group shells by rounding b-values to the nearest 100

## Implementation notes

- The code uses MRtrix shell handling and spherical harmonic helpers.
- The ODF DC term is fixed to the isotropic value.
- The ODF can be modulated by constrasts derived from the Gaussian fractions

## Example workflow

A typical workflow is:

1. Build the module.
2. Run `dwi2fod_lore_sd` with a brain mask.
3. Inspect `odf.mif`, `fracs.mif`, and `response.mif` in an MRtrix viewer.
4. Optionally export `--init_obj_fun` and `--final_obj_fun` for quality checks.

## License

A license file is not included in this repository snapshot.

## Author

Siebe Leysen
