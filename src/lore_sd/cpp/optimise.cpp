#include "optimise.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>

#include <nlopt.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace lore_sd {

// Helper function to get number of SH coefficients
int n4l(int lmax) {
    return (lmax + 1) * (lmax + 2) / 2;
}

// The objective function for nlopt
double objective_function(const std::vector<double> &x, std::vector<double> &grad, void *data) {
    OptData* d = reinterpret_cast<OptData*>(data);
    
    auto S_buf = d->S.request();
    auto gaussians_buf = d->gaussians.request();

    double* S_ptr = static_cast<double*>(S_buf.ptr);
    double* gaussians_ptr = static_cast<double*>(gaussians_buf.ptr);

    int n_lmax = n4l(d->lmax);
    int n_gauss = d->n_gaussians;
    int n_shells = d->S.shape(0);
    int n_rh = d->gaussians.shape(2);

    const double* odf = x.data();
    const double* fs = x.data() + n_lmax;

    // Reconstruct response function (kernel)
    std::vector<double> kernel(n_shells * n_rh, 0.0);
    for (int i = 0; i < n_shells; ++i) {
        for (int j = 0; j < n_rh; ++j) {
            for (int k = 0; k < n_gauss; ++k) {
                kernel[i * n_rh + j] += fs[k] * gaussians_ptr[k * (n_shells * n_rh) + i * n_rh + j];
            }
        }
    }

    // Spherical convolution
    std::vector<double> convolved(n_shells * n_lmax, 0.0);
     for (int i = 0; i < n_shells; ++i) {
        for (int l_idx = 0; l_idx < n_lmax; ++l_idx) {
            int l = 0, m = 0;
            // This is a simplified way to get l from l_idx. A more robust solution is needed.
            int temp_idx = 0;
            for(int l_val = 0; l_val <= d->lmax; l_val+=2){
                for(int m_val = -l_val; m_val <= l_val; ++m_val){
                    if(temp_idx == l_idx) {
                        l = l_val;
                        m = m_val;
                        break;
                    }
                    temp_idx++;
                }
                if(l != 0) break;
            }
            convolved[i * n_lmax + l_idx] = kernel[i * n_rh + l/2] * odf[l_idx];
        }
    }

    // Differences
    std::vector<double> differences(n_shells * n_lmax, 0.0);
    for(int i=0; i< n_shells * n_lmax; ++i) {
        differences[i] = S_ptr[i] - convolved[i];
    }

    // Cost
    double cost = 0.0;
    for (double diff : differences) {
        cost += diff * diff;
    }

    // Regularization
    double reg_term = 0.0;
    for (int i = 0; i < n_shells; ++i) {
        for (int j = 1; j < n_rh; ++j) {
            reg_term += std::pow(kernel[i * n_rh + j], 2);
        }
    }
    cost += d->reg * reg_term;

    // Gradient calculation
    if (!grad.empty()) {
        std::fill(grad.begin(), grad.end(), 0.0);
        // Grad wrt ODF
        for (int l_idx = 0; l_idx < n_lmax; ++l_idx) {
            int l = 0;
            int temp_idx = 0;
            for(int l_val = 0; l_val <= d->lmax; l_val+=2){
                for(int m_val = -l_val; m_val <= l_val; ++m_val){
                    if(temp_idx == l_idx) {
                        l = l_val;
                        break;
                    }
                    temp_idx++;
                }
                if(l != 0) break;
            }
            for (int i = 0; i < n_shells; ++i) {
                grad[l_idx] -= 2.0 * differences[i * n_lmax + l_idx] * kernel[i * n_rh + l/2];
            }
        }

        // Grad wrt FS
        for (int k = 0; k < n_gauss; ++k) {
            for (int i = 0; i < n_shells; ++i) {
                for (int l_idx = 0; l_idx < n_lmax; ++l_idx) {
                     int l = 0;
                    int temp_idx = 0;
                    for(int l_val = 0; l_val <= d->lmax; l_val+=2){
                        for(int m_val = -l_val; m_val <= l_val; ++m_val){
                            if(temp_idx == l_idx) {
                                l = l_val;
                                break;
                            }
                            temp_idx++;
                        }
                        if(l != 0) break;
                    }
                    grad[n_lmax + k] -= 2.0 * differences[i * n_lmax + l_idx] * gaussians_ptr[k * (n_shells * n_rh) + i * n_rh + l/2] * odf[l_idx];
                }
            }
            // Grad of regularization term
            for (int i = 0; i < n_shells; ++i) {
                for (int j = 1; j < n_rh; ++j) {
                    grad[n_lmax + k] += d->reg * 2.0 * kernel[i * n_rh + j] * gaussians_ptr[k * (n_shells * n_rh) + i * n_rh + j];
                }
            }
        }
        
        for(size_t i = 0; i < grad.size(); ++i) {
            grad[i] *= 1e-5;
        }
    }

    return 1e-5 * cost;
}

void sum_of_fractions_constraint(unsigned m, double *result, unsigned n, const double* x, double* grad, void* data) {
    OptData* d = reinterpret_cast<OptData*>(data);
    int n_lmax = n4l(d->lmax);
    
    double sum = 0.0;
    for (int i = 0; i < d->n_gaussians; ++i) {
        sum += x[n_lmax + i];
    }
    result[0] = sum - 1.0;

    if (grad) {
        for(unsigned i = 0; i < n; ++i) grad[i] = 0.0;
        for (int i = 0; i < d->n_gaussians; ++i) {
            grad[n_lmax + i] = 1.0;
        }
    }
}

void non_negative_odf_constraint(unsigned m, double *result, unsigned n, const double* x, double* grad, void* data) {
    OptData* d = reinterpret_cast<OptData*>(data);
    auto Q_buf = d->Q.request();
    double* Q_ptr = static_cast<double*>(Q_buf.ptr);
    int n_lmax = n4l(d->lmax);
    int q_rows = d->Q.shape(0);

    for (int i = 0; i < q_rows; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < n_lmax; ++j) {
            result[i] += Q_ptr[i * n_lmax + j] * x[j];
        }
    }

    if (grad) {
        for(unsigned i = 0; i < m * n; ++i) grad[i] = 0.0;
        for (int i = 0; i < q_rows; ++i) {
            for (int j = 0; j < n_lmax; ++j) {
                grad[i * n + j] = Q_ptr[i * n_lmax + j];
            }
        }
    }
}


Optimiser::Optimiser(int lmax, int n_gaussians, py::array_t<double> S, py::array_t<double> gaussians, py::array_t<double> Q, double reg) {
    data.lmax = lmax;
    data.n_gaussians = n_gaussians;
    data.S = S;
    data.gaussians = gaussians;
    data.Q = Q;
    data.reg = reg;
}

std::vector<double> Optimiser::optimise(std::vector<double>& x) {
    int n_lmax = n4l(data.lmax);
    int n_params = n_lmax + data.n_gaussians;

    nlopt::opt opt(nlopt::LD_SLSQP, n_params);
    opt.set_min_objective(objective_function, &data);

    // Bounds
    std::vector<double> lb(n_params);
    std::vector<double> ub(n_params);
    lb[0] = 1.0 / std::sqrt(4.0 * M_PI);
    ub[0] = 1.0 / std::sqrt(4.0 * M_PI);
    for(int i = 1; i < n_lmax; ++i) {
        lb[i] = -1e10; // Using a large number for infinity
        ub[i] = 1e10;
    }
    for(int i = 0; i < data.n_gaussians; ++i) {
        lb[n_lmax + i] = 0.0;
        ub[n_lmax + i] = 1.0;
    }
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    // Constraints
    std::vector<double> tol_eq(1, 1e-8);
    opt.add_equality_mconstraint(sum_of_fractions_constraint, &data, tol_eq);

    int q_rows = data.Q.shape(0);
    std::vector<double> tol_ineq(q_rows, 1e-8);
    opt.add_inequality_mconstraint(non_negative_odf_constraint, &data, tol_ineq);

    opt.set_xtol_rel(1e-6);
    opt.set_maxeval(1000);

    double minf;
    try {
        nlopt::result result = opt.optimize(x, minf);
        (void)result;
    } catch (const std::exception &e) {
        std::string msg = e.what();
        if (msg.find("roundoff") == std::string::npos) {
            std::cerr << "nlopt failed: " << e.what() << std::endl;
        }
    }

    return x;
}

} // namespace lore_sd
