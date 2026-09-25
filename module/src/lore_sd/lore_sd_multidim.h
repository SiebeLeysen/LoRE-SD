#ifndef LORE_SD_MRTRIX3_LORE_SD_MULTIDIM_H
#define LORE_SD_MRTRIX3_LORE_SD_MULTIDIM_H

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace LoreSD {

// Configuration passed into the voxel fitter.
// The command wrapper populates these fields from MRtrix CLI options and the
// diffusion image metadata.
struct Params {
  int lmax;
  double reg = 4e-5;
  int grid_size[3] = {10, 10, 10};
  bool init_obj_fun = false;
  bool final_obj_fun = false;
  std::string debug_dump;
  std::vector<double> bvals;
  std::vector<double> beta;
  std::vector<double> te;
  std::vector<double> t2;
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
  std::vector<float> model_weights;
  std::vector<float> response;
  std::vector<float> predicted_signal;
  double f0 = 0.0;
  double f1 = 0.0;
  int status = 0;
};

// Build a reusable parameter bundle from the gradient table and shell layout.
Params make_params_multidim(int lmax,
                   const int grid_size[3],
                   double reg,
                   const Eigen::MatrixXd& grad,
                   const Eigen::MatrixXd& eval_dirs,
                   const std::vector<double>& bvals,
                   const std::vector< std::vector<size_t> >& shell_volumes,
                   const std::vector<double>& beta = {},
                   const std::vector<double>& te = {});

// Fit one voxel and return the LoRE-SD outputs.
Result fit_voxel_multidim(const Eigen::VectorXd& dwi, const Params& params);

// Predict measurements for an arbitrary acquisition parameterization.
void predict_measurements(const Eigen::MatrixXd& rf,
                          const Eigen::VectorXd& odf,
                          const Params& prediction_params,
                          Eigen::VectorXd& predicted);

// Convenience wrapper: predict from packed fit output response/odf.
void predict_from_fit_result(const Result& fit_result,
                             const Params& prediction_params,
                             Eigen::VectorXd& predicted);

}

#endif
