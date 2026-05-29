#include "lore_sd/lore_sd.h"

#include <cmath>
#include <limits>
#include <iostream>

#include <nlopt.hpp>

#include "math/math.h"
#include "math/SH.h"
#include "math/constrained_least_squares.h"
#include "math/legendre.h"
#include "math/least_squares.h"
#include "math/sphere.h"

#include <Eigen/Dense>

namespace LoreSD
{

  namespace
  {

    struct Workspace;

    struct OptData
    {
      Eigen::MatrixXd S;
      const std::vector<Eigen::MatrixXd> *gaussians_rh;

      Eigen::MatrixXd gaussians_rh_flat;

      Eigen::MatrixXd Q;
      double reg;
      int lmax;
      int n_sh;
      int n_gauss;
      int n_shells;
      int n_rh;
      const double *odf_ptr = nullptr;


      // Reusable buffers (mutable because objective is logically const)
      mutable Eigen::VectorXd buf_kernel; // length M
      mutable Eigen::VectorXd buf_diff;   // length M
      mutable Eigen::VectorXd buf_tmp;    // length M
      mutable Eigen::VectorXd buf_reg;    // length M (only used if reg enabled)
      // no profiling counters in public release

    };

    struct CSDData
    {
      Eigen::MatrixXd S;
      Eigen::MatrixXd rf;
      Eigen::MatrixXd Q;
      int n_sh;
    };

    inline int n4l(int lmax)
    {
      return static_cast<int>(MR::Math::SH::NforL(lmax));
    }

    Eigen::MatrixXd expcoefs(const Eigen::VectorXd &a, int lmax)
    {
      const int ncoeff = lmax / 2 + 1;
      const int nx = 10 * ncoeff + 1;
      Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(nx, -1.0, 1.0);
      x = x.tail(nx - 1);

      Eigen::MatrixXd P(x.size(), ncoeff);
      for (int i = 0; i < x.size(); ++i)
      {
        for (int l = 0, col = 0; l <= lmax; l += 2, ++col)
        {
          P(i, col) = MR::Math::Legendre::Plm(l, 0, x[i]);
        }
      }

      Eigen::MatrixXd coefs(a.size(), ncoeff);
      for (int i = 0; i < a.size(); ++i)
      {
        Eigen::VectorXd y = (-a[i] * x.array().square()).exp();
        Eigen::VectorXd coef = P.colPivHouseholderQr().solve(y);
        coefs.row(i) = coef.transpose();
      }

      return coefs;
    }

    Eigen::MatrixXd zhgaussian(const std::vector<double> &bvals, double Da, double Dr, int lmax)
    {
      const int n_shells = static_cast<int>(bvals.size());
      const int ncoeff = lmax / 2 + 1;

      Eigen::VectorXd b = Eigen::Map<const Eigen::VectorXd>(bvals.data(), n_shells);
      Eigen::VectorXd a = b.array() * (Da - Dr);
      Eigen::MatrixXd expc = expcoefs(a, lmax);

      Eigen::VectorXd scale(ncoeff);
      for (int l = 0, col = 0; l <= lmax; l += 2, ++col)
      {
        scale[col] = std::sqrt(4.0 * MR::Math::pi / (2.0 * l + 1.0));
      }

      Eigen::MatrixXd gauss(n_shells, ncoeff);
      for (int i = 0; i < n_shells; ++i)
      {
        for (int j = 0; j < ncoeff; ++j)
        {
          gauss(i, j) = scale[j] * std::exp(-b[i] * Dr) * expc(i, j);
        }
      }

      if (gauss(0, 0) != 0.0)
        gauss.array() /= gauss(0, 0);

      return gauss;
    }

    Eigen::MatrixXd zh2rh(const Eigen::MatrixXd &zh, int lmax)
    {
      const int ncoeff = lmax / 2 + 1;
      const int n_sh = n4l(lmax);
      Eigen::MatrixXd z2r = Eigen::MatrixXd::Zero(ncoeff, n_sh);
      for (int l = 0; l <= lmax; l += 2)
      {
        const int j1 = n4l(l - 2);
        const int j2 = n4l(l);
        const double scale = std::sqrt(4.0 * MR::Math::pi / (2.0 * l + 1.0));
        for (int j = j1; j < j2; ++j)
        {
          z2r(l / 2, j) = scale;
        }
      }
      return zh * z2r;
    }

    Eigen::MatrixXd rh2zh(const Eigen::MatrixXd &rh, int lmax)
    {
      const int ncoeff = lmax / 2 + 1;
      Eigen::MatrixXd zh(rh.rows(), ncoeff);
      int idx = 0;
      for (int l = 0; l <= lmax; l += 2)
      {
        const double scale = 1.0 / std::sqrt(4.0 * MR::Math::pi / (2.0 * l + 1.0));
        zh.col(l / 2) = scale * rh.col(idx);
        idx += 2 * l + 1;
      }
      return zh;
    }

    // Precompute the Gaussian basis in real SH form for every allowed (Da, Dr) pair.
    // The outer grid is square; invalid points (Da < Dr) are kept at zero.
    std::vector<Eigen::MatrixXd> compute_gaussians_rh(const Params &params)
    {
      const int n_shells = static_cast<int>(params.bvals.size());
      const int ncoeff = params.lmax / 2 + 1;
      const int n_gauss = params.grid_size * params.grid_size;

      Eigen::MatrixXd gauss_zh(n_gauss, n_shells * ncoeff);
      gauss_zh.setZero();

      int idx = 0;
      for (double da : params.da)
      {
        for (double dr : params.dr)
        {
          if (da >= dr)
          {
            Eigen::MatrixXd zh = zhgaussian(params.bvals, da, dr, params.lmax);
            for (int s = 0; s < n_shells; ++s)
            {
              gauss_zh.block(idx, s * ncoeff, 1, ncoeff) = zh.row(s);
            }
          }
          ++idx;
        }
      }

      std::vector<Eigen::MatrixXd> gauss_rh;
      gauss_rh.reserve(n_gauss);

      for (int g = 0; g < n_gauss; ++g)
      {
        Eigen::MatrixXd zh(n_shells, ncoeff);
        for (int s = 0; s < n_shells; ++s)
        {
          zh.row(s) = gauss_zh.block(g, s * ncoeff, 1, ncoeff);
        }
        Eigen::MatrixXd rh = zh2rh(zh, params.lmax);
        gauss_rh.push_back(rh);
      }

      return gauss_rh;
    }

    void compute_signal_sh(const Eigen::VectorXd &dwi,
                           const Params &params,
                           Eigen::MatrixXd &S,
                           Eigen::VectorXd &shell_signal)
    {
      const int n_shells = static_cast<int>(params.shell_pinvQ.size());
      const int n_sh = n4l(params.lmax);
      if (S.rows() != n_shells || S.cols() != n_sh)
        S.resize(n_shells, n_sh);
      S.setZero();

      for (int s = 0; s < n_shells; ++s)
      {
        const auto &indices = params.shell_volumes[s];
        const int ncoeff = params.shell_ncoeff[s];
        const auto &pinv = params.shell_pinvQ[s];

        shell_signal.resize(indices.size());
        for (size_t i = 0; i < indices.size(); ++i)
          shell_signal[i] = dwi[indices[i]];

        Eigen::VectorXd sh = pinv * shell_signal;
        S.block(s, 0, 1, ncoeff) = sh.transpose();
      }
    }

    // Build the initial parameter vector and bounds.
    // ODF DC is fixed to the isotropic value and fractions are initialized uniformly.
    void init_bounds(const Params &params, std::vector<double> &x0, std::vector<double> &lb, std::vector<double> &ub)
    {
      const int n_sh = n4l(params.lmax);
      const int n_gauss = params.grid_size * params.grid_size;

      x0.assign(n_sh + n_gauss, 0.0);
      lb.assign(n_sh + n_gauss, -std::numeric_limits<double>::infinity());
      ub.assign(n_sh + n_gauss, std::numeric_limits<double>::infinity());

      const double dc = 1.0 / std::sqrt(4.0 * MR::Math::pi);
      x0[0] = dc;
      lb[0] = dc;
      ub[0] = dc;

      int idx = 0;
      int valid = 0;
      for (double da : params.da)
      {
        for (double dr : params.dr)
        {
          if (da >= dr)
            ++valid;
        }
      }
      const double init_frac = valid > 0 ? (1.0 / static_cast<double>(valid)) : 0.0;
      for (double da : params.da)
      {
        for (double dr : params.dr)
        {
          const bool allowed = da >= dr;
          lb[n_sh + idx] = allowed ? 0.0 : 0.0;
          ub[n_sh + idx] = allowed ? 1.0 : 0.0;
          x0[n_sh + idx] = allowed ? init_frac : 0.0;
          ++idx;
        }
      }
    }

    void sum_of_fractions_constraint(unsigned, double *result, unsigned n, const double *x, double *grad, void *data)
    {
      auto *d = reinterpret_cast<OptData *>(data);
      double sum = 0.0;
      for (int i = 0; i < d->n_gauss; ++i)
        sum += x[d->n_sh + i];
      result[0] = sum - 1.0;
      if (grad)
      {
        for (unsigned i = 0; i < n; ++i)
          grad[i] = 0.0;
        for (int i = 0; i < d->n_gauss; ++i)
          grad[d->n_sh + i] = 1.0;
      }
    }

    void non_negative_odf_constraint(unsigned m, double *result, unsigned n, const double *x, double *grad, void *data)
    {
      auto *d = reinterpret_cast<OptData *>(data);
      Eigen::Map<const Eigen::VectorXd> odf_map(x, d->n_sh);
      Eigen::Map<Eigen::VectorXd> res(result, d->Q.rows());
      res.noalias() = -d->Q * odf_map;
      if (grad)
      {
        for (unsigned i = 0; i < m * n; ++i)
          grad[i] = 0.0;
        for (int i = 0; i < d->Q.rows(); ++i)
        {
          for (int j = 0; j < d->n_sh; ++j)
            grad[i * n + j] = -d->Q(i, j);
        }
      }
    }

    // Main coupled objective over ODF and Gaussian fractions.
    // The implementation is written to minimize allocations and reuse buffers.
    double objective_function(const std::vector<double>& x,
          std::vector<double>& grad,
          void* data)
    {
      // objective without profiling timestamps
        constexpr double kScale = 1e-5;

        auto* d = reinterpret_cast<OptData*>(data);

        const int n_sh     = d->n_sh;
        const int n_gauss  = d->n_gauss;
        const int n_shells = d->n_shells;

        const int M = n_sh * n_shells;

        const bool compute_grad = !grad.empty();

        // -------------------------------------------------------
        // Views
        // -------------------------------------------------------

        const double* x_ptr = x.data();

        const double* odf = x_ptr;
        const double* fs  = x_ptr + n_sh;

        const Eigen::Map<const Eigen::VectorXd> fs_vec(fs, n_gauss);

        const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
            G(d->gaussians_rh_flat.data(), n_gauss, M);

        const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
            S(d->S.data(), n_shells, n_sh);

        // -------------------------------------------------------
        // Buffers (assumed preallocated in OptData ideally)
        // -------------------------------------------------------

        Eigen::VectorXd& K = d->buf_kernel;
        Eigen::VectorXd& D = d->buf_diff;
        Eigen::VectorXd& T = d->buf_tmp;

        if ((int)K.size() != M) { K.resize(M); D.resize(M); T.resize(M); }

        // -------------------------------------------------------
        // KERNEL: K = G^T fs  (as matrix Kmat of size n_shells x n_sh)
        // -------------------------------------------------------

        K.noalias() = G.transpose() * fs_vec;

        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> Kmat(K.data(), n_shells, n_sh);

        // Compute diff matrix: diff = S - Kmat * diag(odf)
        Eigen::MatrixXd diff = S;
        for (int j = 0; j < n_sh; ++j)
        {
          diff.col(j).noalias() -= Kmat.col(j) * odf[j];
        }

        double cost = diff.squaredNorm();

        // write D and T buffers (column-major mapping)
        // D = diff (flattened), T = diff .* odf_rep (elementwise multiplied by odf per column)
        if ((int)D.size() != M) { D.resize(M); T.resize(M); }
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> Dmat(D.data(), n_shells, n_sh);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> Tmat(T.data(), n_shells, n_sh);
        Dmat = diff;
        for (int j = 0; j < n_sh; ++j)
          Tmat.col(j) = diff.col(j) * odf[j];

        // -------------------------------------------------------
        // REGULARIZATION (kernel-based, masked view)
        // -------------------------------------------------------

        if (d->reg > 0.0 && n_sh > 1 && n_shells > 1)
        {
          for (int j = 1; j < n_sh; ++j)
          {
            // sum squares of Kmat entries excluding i=0
            cost += d->reg * Kmat.col(j).segment(1, n_shells - 1).squaredNorm();
          }
        }

        // -------------------------------------------------------
        // GRADIENT
        // -------------------------------------------------------

        if (compute_grad)
        {
            Eigen::Map<Eigen::VectorXd> g(grad.data(), grad.size());
            g.setZero();

            // =====================================================
            // d/d odf  (fully fused dot product)
            // =====================================================

            for (int j = 0; j < n_sh; ++j)
            {
              // g[j] = -2 * Kj' * Dj
              g[j] = -2.0 * Kmat.col(j).dot(diff.col(j));
            }

            // =====================================================
            // fs gradient: GEMV (fast path)
            // =====================================================

            // fs gradient: GEMV (fast path)
            Eigen::Map<const Eigen::VectorXd> Tvec(T.data(), M);
            g.segment(n_sh, n_gauss).noalias() = -2.0 * d->gaussians_rh_flat * Tvec;

            // =====================================================
            // regularization gradient (fs only, GEMV form)
            // =====================================================

            if (d->reg > 0.0 && n_sh > 1 && n_shells > 1)
            {
              // build masked kernel in-place using Dmat as temp: Rj[0]=0, Rj[i]=Kj[i] for i>=1
              Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> Rmat(D.data(), n_shells, n_sh);
              for (int j = 1; j < n_sh; ++j)
              {
                Rmat(0, j) = 0.0;
                Rmat.col(j).segment(1, n_shells - 1) = Kmat.col(j).segment(1, n_shells - 1);
              }

              g.segment(n_sh, n_gauss).noalias() +=
                (2.0 * d->reg) *
                d->gaussians_rh_flat *
                Eigen::Map<const Eigen::VectorXd>(Rmat.data(), M);
            }

            g *= kScale;
        }
          // std::cout << "Cost: " << cost << std::endl;
              return kScale * cost;
    }

    struct Workspace
    {
      bool initialized = false;
      int n_sh = 0;
      int n_gauss = 0;
      int n_shells = 0;
      int n_rh = 0;
      std::vector<double> x0;
      std::vector<double> lb;
      std::vector<double> ub;
      std::vector<double> grad_vec;
      std::vector<double> x_init;
      std::vector<Eigen::MatrixXd> gaussians_scaled;
      Eigen::MatrixXd S;
      Eigen::MatrixXd rf;
      Eigen::MatrixXd kernel;
      Eigen::MatrixXd diff;
      Eigen::VectorXd odf_grad;
      
      Eigen::MatrixXd gaussians_rh_flat;
      Eigen::VectorXd shell_signal;
      OptData data;
      CSDData csd;
      nlopt::opt opt_main;

      void init(const Params &params)
      {
        const int n_sh_local = n4l(params.lmax);
        const int n_gauss_local = params.grid_size * params.grid_size;
        const int n_shells_local = static_cast<int>(params.bvals.size());
        if (initialized && n_sh == n_sh_local && n_gauss == n_gauss_local && n_shells == n_shells_local)
          return;

        initialized = true;
        n_sh = n_sh_local;
        n_gauss = n_gauss_local;
        n_shells = n_shells_local;
        n_rh = params.lmax / 2 + 1;

        x0.resize(n_sh + n_gauss);
        lb.resize(n_sh + n_gauss);
        ub.resize(n_sh + n_gauss);
        grad_vec.resize(n_sh + n_gauss);
        S.resize(n_shells, n_sh);
        rf.resize(n_shells, n_sh);
        kernel.resize(n_shells, n_sh);
        diff.resize(n_shells, n_sh);
        odf_grad.resize(n_sh);
        
        gaussians_rh_flat.resize(n_gauss, n_shells * n_sh);

        gaussians_scaled.resize(n_gauss);

        data.Q = params.Q;
        data.reg = params.reg;
        data.lmax = params.lmax;
        data.n_sh = n_sh;
        data.n_gauss = n_gauss;
        data.n_shells = n_shells;
        data.n_rh = n_rh;
        data.gaussians_rh = &gaussians_scaled;

        csd.Q = params.Q;
        csd.n_sh = n_sh;

        // Set up AUGLAG with LD_LBFGS local solver (lower overhead than SLSQP)
        nlopt::opt local_opt(nlopt::LD_LBFGS, n_sh + n_gauss);
        local_opt.set_xtol_rel(1e-7);
        local_opt.set_ftol_rel(1e-7);

        opt_main = nlopt::opt(nlopt::LD_AUGLAG, n_sh + n_gauss);
        opt_main.set_local_optimizer(local_opt);
        opt_main.set_min_objective(objective_function, &data);
        opt_main.set_maxeval(params.maxeval);

        std::vector<double> tol_eq(1, 1e-2);
        opt_main.add_equality_mconstraint(sum_of_fractions_constraint, &data, tol_eq);
        std::vector<double> tol_ineq(params.Q.rows(), 1e-2);
        opt_main.add_inequality_mconstraint(non_negative_odf_constraint, &data, tol_ineq);
      }
    };

    static void solve_odf_icls(Eigen::MatrixXd &rf,
                               Eigen::MatrixXd &S,
                               std::vector<double> &x0,
                               const Params &params,
                               int n_sh,
                               int n_shells)
    {
      const double dc = 1.0 / std::sqrt(4.0 * MR::Math::pi);

      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_shells * n_sh, n_sh);
      Eigen::VectorXd b = Eigen::VectorXd::Zero(n_shells * n_sh);

      for (int j = 0; j < n_sh; ++j)
      {
        for (int i = 0; i < n_shells; ++i)
        {
          const int row = j * n_shells + i;
          H(row, j) = rf(i, j);
          b(row) = S(i, j);
        }
      }

      Eigen::MatrixXd A = params.Q;
      Eigen::VectorXd t = Eigen::VectorXd::Zero(A.rows());
      Eigen::MatrixXd B = Eigen::MatrixXd::Zero(1, n_sh);
      Eigen::VectorXd s = Eigen::VectorXd::Zero(1);
      B(0, 0) = 1.0;
      s[0] = dc;

      auto problem = MR::Math::ICLS::Problem<double>(H, A, B, t, s, 1e-10, 1e-10, 30, 1e-10);
      auto solver = MR::Math::ICLS::Solver<double>(problem);

      Eigen::VectorXd odf = Eigen::Map<Eigen::VectorXd>(x0.data(), n_sh);
      const size_t niter = solver(odf, b);
      (void)niter;

      for (int i = 0; i < n_sh; ++i)
        x0[i] = odf[i];
      x0[0] = dc;
    }

  } // namespace

  // Construct all static geometry, basis and shell-precomputation data once.
  Params make_params(int lmax,
                     int grid_size,
                     double reg,
                     const Eigen::MatrixXd &grad,
                     const Eigen::MatrixXd &eval_dirs,
                     const std::vector<double> &bvals,
                     const std::vector<std::vector<size_t>> &shell_volumes)
  {
    Params params;
    params.lmax = lmax;
    params.grid_size = grid_size;
    params.reg = reg;
    params.bvals = bvals;

    params.da.resize(grid_size);
    params.dr.resize(grid_size);

    const double max_diffusivity = 4e-3;
    if (grid_size <= 1)
    {
      params.da[0] = 0.0;
      params.dr[0] = 0.0;
    }
    else
    {
      const double step = max_diffusivity / static_cast<double>(grid_size - 1);
      for (int i = 0; i < grid_size; ++i)
      {
        params.da[i] = step * static_cast<double>(i);
        params.dr[i] = step * static_cast<double>(i);
      }
    }

    const int n_shells = static_cast<int>(shell_volumes.size());
    params.shell_sizes.resize(n_shells);
    params.shell_ncoeff.resize(n_shells);
    params.shell_Q.resize(n_shells);
    params.shell_pinvQ.resize(n_shells);
    params.shell_volumes = shell_volumes;

    const Eigen::MatrixXd grad_dirs = grad.leftCols(3);
    if (eval_dirs.rows() > 0)
    {
      params.Q = MR::Math::SH::init_transform(eval_dirs, lmax);
    }
    else
    {
      params.Q = MR::Math::SH::init_transform_cart(grad_dirs, lmax);
    }

    for (int s = 0; s < n_shells; ++s)
    {
      const auto &vols = shell_volumes[s];
      params.shell_sizes[s] = static_cast<int>(vols.size());

      int nn = 1;
      if (!vols.empty())
      {
        if (bvals[s] > 10.0)
        {
          for (int l = 0; l <= lmax; l += 2)
          {
            int ncoeff = n4l(l);
              if (ncoeff < static_cast<int>(vols.size()))
              nn = ncoeff;
          }
        }
      }
      params.shell_ncoeff[s] = nn;

      Eigen::MatrixXd dirs(vols.size(), 3);
      for (size_t i = 0; i < vols.size(); ++i)
        dirs.row(i) = grad_dirs.row(vols[i]);

      Eigen::MatrixXd Q = MR::Math::SH::init_transform_cart(dirs, lmax);
      Eigen::MatrixXd Qnn = Q.leftCols(nn);
      params.shell_Q[s] = Qnn;
      params.shell_pinvQ[s] = MR::Math::pinv(Qnn);
    }

    params.gaussians_rh = compute_gaussians_rh(params);

    return params;
  }

  // Fit one voxel. This performs the per-voxel solve and packages outputs.
  Result fit_voxel(const Eigen::VectorXd &dwi, const Params &params)
  {
    Result result;

      static thread_local Workspace ws;
    ws.init(params);

    const int n_sh = n4l(params.lmax);
    const int n_gauss = params.grid_size * params.grid_size;
    const int n_shells = static_cast<int>(params.bvals.size());

    result.odf.assign(n_sh, 0.0f);
    result.fracs.assign(n_gauss, 0.0f);
    result.response.assign(n_shells * (params.lmax / 2 + 1), 0.0f);
    result.predicted_signal.assign(static_cast<size_t>(dwi.size()), 0.0f);

    if (dwi.size() == 0)
      return result;

    compute_signal_sh(dwi, params, ws.S, ws.shell_signal);
    if (ws.S(0, 0) == 0.0)
      return result;

    const double s00 = ws.S(0, 0);
    double scale_factor = 1000.0 / s00;
    ws.S *= scale_factor;
    const double s00_scaled = ws.S(0, 0);

    for (int g = 0; g < ws.n_gauss; ++g)
    {
      ws.gaussians_scaled[g] = params.gaussians_rh[g];
      ws.gaussians_scaled[g] *= s00_scaled;
      Eigen::Map<const Eigen::RowVectorXd> flat(ws.gaussians_scaled[g].data(), ws.gaussians_scaled[g].size());

      ws.gaussians_rh_flat.row(g) = flat;
    }
    ws.data.gaussians_rh_flat = ws.gaussians_rh_flat;

    init_bounds(params, ws.x0, ws.lb, ws.ub);

    Eigen::VectorXd fs_init = Eigen::Map<Eigen::VectorXd>(ws.x0.data() + n_sh, n_gauss);
    ws.rf.setZero();
    for (int g = 0; g < n_gauss; ++g)
      ws.rf += fs_init[g] * ws.gaussians_scaled[g];

    ws.csd.S = ws.S;
    ws.csd.rf = ws.rf;
    std::vector<double> lb_csd(n_sh, -std::numeric_limits<double>::infinity());
    std::vector<double> ub_csd(n_sh, std::numeric_limits<double>::infinity());
    const double dc = 1.0 / std::sqrt(4.0 * MR::Math::pi);
    lb_csd[0] = dc;
    ub_csd[0] = dc;
    // ws.opt_csd.set_lower_bounds(lb_csd);
    // ws.opt_csd.set_upper_bounds(ub_csd);
    

    std::vector<double> odf_init(ws.x0.begin(), ws.x0.begin() + n_sh);
    const Eigen::VectorXd odf_init_before = Eigen::Map<const Eigen::VectorXd>(odf_init.data(), n_sh);

    try
    {
      solve_odf_icls(ws.rf, ws.S, ws.x0, params, ws.n_sh, ws.n_shells);
    }
    catch (const std::exception &)
    {
    }
    const Eigen::Map<const Eigen::VectorXd> odf_init_after(ws.x0.data(), n_sh);

    // set up OptData for subsequent steps
    ws.data.S = ws.S;
    ws.data.gaussians_rh = &ws.gaussians_scaled;
    ws.data.gaussians_rh_flat = ws.gaussians_rh_flat;
    ws.data.odf_ptr = ws.x0.data();

    // Main optimization (single constrained optimization over odf+fractions)
    ws.opt_main.set_lower_bounds(ws.lb);
    ws.opt_main.set_upper_bounds(ws.ub);

    // ensure internal data views are set correctly
    ws.data.S = ws.S;
    ws.data.gaussians_rh = &ws.gaussians_scaled;
    ws.data.gaussians_rh_flat = ws.gaussians_rh_flat;
    ws.data.odf_ptr = nullptr; // objective will read odf from x when needed

    double minf = 0.0;
    const std::vector<double> x_init = ws.x0;
    nlopt::result status = nlopt::FAILURE;
    try
    {
      status = ws.opt_main.optimize(ws.x0, minf);
    }
    catch (const std::exception &)
    {
      // optimization failed - fall back to current x0
    }

    Eigen::VectorXd odf = Eigen::Map<Eigen::VectorXd>(ws.x0.data(), n_sh);
    Eigen::VectorXd fs = Eigen::Map<Eigen::VectorXd>(ws.x0.data() + n_sh, n_gauss);

    Eigen::VectorXd kernel_flat = ws.gaussians_rh_flat.transpose() * fs;
    Eigen::Map<const Eigen::MatrixXd> kernel(kernel_flat.data(), n_shells, n_sh);
    ws.kernel = kernel;

    Eigen::MatrixXd response = rh2zh(ws.kernel, params.lmax) / scale_factor;

    Eigen::MatrixXd response_rh = zh2rh(response, params.lmax);
    Eigen::VectorXd predicted = Eigen::VectorXd::Zero(static_cast<int>(dwi.size()));
    for (int s = 0; s < n_shells; ++s)
    {
      const auto &indices = params.shell_volumes[s];
      const int nn = params.shell_ncoeff[s];
      if (indices.empty() || nn <= 0)
        continue;

      Eigen::VectorXd shell_coeffs = response_rh.row(s).transpose().cwiseProduct(odf);
      Eigen::VectorXd shell_pred = params.shell_Q[s] * shell_coeffs.head(nn);
      for (size_t i = 0; i < indices.size(); ++i)
        predicted[static_cast<int>(indices[i])] = shell_pred[static_cast<int>(i)];
    }

    result.odf.assign(odf.data(), odf.data() + odf.size());
    result.fracs.assign(fs.data(), fs.data() + fs.size());
    result.response.assign(response.data(), response.data() + response.size());
    result.predicted_signal.assign(predicted.data(), predicted.data() + predicted.size());

    const bool need_objectives = params.init_obj_fun || params.final_obj_fun;
    if (need_objectives)
    {
      std::vector<double> grad_tmp(ws.x0.size(), 0.0);
      double f0 = objective_function(x_init, grad_tmp, &ws.data);
      double f1 = objective_function(ws.x0, grad_tmp, &ws.data);

      if (!std::isfinite(f0))
        f0 = -1.0;
      if (!std::isfinite(f1))
        f1 = -1.0;

      result.f0 = f0;
      result.f1 = f1;
    }

    result.status = static_cast<int>(status);

    return result;
  }
}
