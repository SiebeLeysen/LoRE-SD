#ifndef LORE_SD_MRTRIX3_LORE_SD_H
#define LORE_SD_MRTRIX3_LORE_SD_H

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace LoreSD {

// Configuration passed into the voxel fitter.
// The command wrapper populates these fields from MRtrix CLI options and the
// diffusion image metadata.
struct Params {
  int lmax;
  double reg = 1e-3;
  int grid_size = 10;
  bool init_obj_fun = false;
  bool final_obj_fun = false;
  std::string debug_dump;
  std::vector<double> bvals;
  std::vector<double> da;
  std::vector<double> dr;
  std::vector<int> valid_gaussian_indices;
  int n_valid_gauss = 0;
  std::vector< Eigen::MatrixXd > gaussians_rh;
  std::vector<int> shell_sizes;
  std::vector<int> shell_ncoeff;
  std::vector< Eigen::MatrixXd > shell_Q;
  std::vector< Eigen::MatrixXd > shell_pinvQ;
  std::vector< std::vector<size_t> > shell_volumes;
  Eigen::MatrixXd Q;
  int max_admm_iter = 300;
  int max_admm_iter_per_als = 60; // If > 0, limits ADMM iterations during intermediate ALS steps
  
};

// Per-voxel outputs returned by the fitter.
// `odf`, `fracs`, `response`, and `predicted_signal` are written directly to MRtrix images.
// `f0` and `f1` are optional objective values for the initial and final states.
struct Result {
  std::vector<float> odf;
  std::vector<float> fracs;
  std::vector<float> response;
  std::vector<float> predicted_signal;
  double f0 = 0.0;
  double f1 = 0.0;
  int status = 0;
};

// Build a reusable parameter bundle from the gradient table and shell layout.
Params make_params(int lmax,
                   int grid_size,
                   double reg,
                   const Eigen::MatrixXd& grad,
                   const Eigen::MatrixXd& eval_dirs,
                   const std::vector<double>& bvals,
                   const std::vector< std::vector<size_t> >& shell_volumes);

// Fit one voxel and return the LoRE-SD outputs.
Result fit_voxel(const Eigen::VectorXd& dwi, const Params& params);

}

#endif
