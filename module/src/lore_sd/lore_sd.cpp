#include "lore_sd/lore_sd.h"

#include <cmath>
#include <algorithm>

#include "math/math.h"
#include "math/SH.h"
#include "math/legendre.h"
#include "math/least_squares.h"
#include "math/sphere.h"

#include <Eigen/Dense>

namespace LoreSD
{
  namespace
  {

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
      
      const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(P);

      Eigen::MatrixXd coefs(a.size(), ncoeff);
      for (int i = 0; i < a.size(); ++i)
      {
        Eigen::VectorXd y = (-a[i] * x.array().square()).exp();
        coefs.row(i) = qr.solve(y).transpose();
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

    struct Workspace
    {
      bool initialized = false;
      int n_sh = 0;
      int n_gauss = 0;
      int n_valid_gauss = 0;
      int n_shells = 0;
      int nf = 0;             // n_sh - 1: "free" (l>0) ODF coefficients
      int n_constraints = 0;  // rows of params.Q

      std::vector<double> x0;

      Eigen::MatrixXd S;
      Eigen::MatrixXd rf;
      Eigen::MatrixXd kernel;

      Eigen::MatrixXd R_base;
      Eigen::MatrixXd RtR;

      // -- fraction ADMM (solve_fractions_admm) --
      Eigen::MatrixXd H_data;
      Eigen::MatrixXd AtA;
      Eigen::VectorXd Atb;
      Eigen::MatrixXd frac_M;
      Eigen::LLT<Eigen::MatrixXd> frac_llt;
      Eigen::VectorXd frac_x, frac_z, frac_u, frac_xhat, frac_rhs, frac_zprev;

      // -- ODF ADMM (solve_odf_admm): quantities derived purely from
      // params.Q, constant across every voxel for a given Params, so
      // they are computed once here instead of once per voxel/outer-iter. --
      Eigen::MatrixXd QTQ_free;   // nf x nf
      Eigen::VectorXd QTQ_dc;     // nf
      Eigen::MatrixXd Q_free;     // n_constraints x nf

      // -- ODF ADMM per-voxel working buffers --
      Eigen::MatrixXd odf_K;
      Eigen::LDLT<Eigen::MatrixXd> odf_ldlt;
      Eigen::VectorXd odf_c, odf_z, odf_u, odf_q, odf_zold, odf_rhs, odf_cfree, odf_admm_free;

      // -- rf_objective scratch --
      Eigen::MatrixXd obj_rf;
      Eigen::MatrixXd obj_pred;

      Eigen::VectorXd prev_odf;
      Eigen::VectorXd prev_frac;

      // -- fraction ADMM persistent state across ALS iterations --
      bool frac_state_valid = false;
      Eigen::VectorXd prev_frac_x;
      Eigen::VectorXd prev_frac_z;
      Eigen::VectorXd prev_frac_u;

      // -- ODF ADMM persistent state across ALS iterations --
      bool odf_state_valid = false;
      Eigen::VectorXd prev_odf_c;
      Eigen::VectorXd prev_odf_z;
      Eigen::VectorXd prev_odf_u;

      Eigen::VectorXd shell_signal;

      void init(const Params &params)
      {
        const int n_sh_local = n4l(params.lmax);
        const int n_gauss_local = params.grid_size * params.grid_size;
        const int n_valid_gauss_local = params.n_valid_gauss;
        const int n_shells_local = static_cast<int>(params.bvals.size());
        if (initialized && n_sh == n_sh_local && n_gauss == n_gauss_local && n_valid_gauss == n_valid_gauss_local && n_shells == n_shells_local)
          return;

        initialized = true;
        n_sh = n_sh_local;
        n_gauss = n_gauss_local;
        n_valid_gauss = n_valid_gauss_local;
        n_shells = n_shells_local;
        nf = n_sh - 1;
        n_constraints = static_cast<int>(params.Q.rows());

        x0.resize(n_sh + n_valid_gauss);
        S.resize(n_shells, n_sh);
        rf.resize(n_shells, n_sh);
        kernel.resize(n_shells, n_sh);

        H_data.resize(n_shells * n_sh, n_valid_gauss);
        AtA.resize(n_valid_gauss, n_valid_gauss);
        Atb.resize(n_valid_gauss);

        frac_M.resize(n_valid_gauss, n_valid_gauss);
        frac_x.resize(n_valid_gauss);
        frac_z.resize(n_valid_gauss);
        frac_u.resize(n_valid_gauss);
        frac_xhat.resize(n_valid_gauss);
        frac_rhs.resize(n_valid_gauss);
        frac_zprev.resize(n_valid_gauss);

        obj_rf.resize(n_shells, n_sh);
        obj_pred.resize(n_shells, n_sh);

        prev_odf.resize(n_sh);

        prev_odf.setZero();
        if (n_sh > 0)
          prev_odf[0] = 1.0 / std::sqrt(4.0 * MR::Math::pi);

        frac_state_valid = false;
        prev_frac_x.resize(n_valid_gauss);
        prev_frac_z.resize(n_valid_gauss);
        prev_frac_u.resize(n_valid_gauss);

        odf_state_valid = false;
        prev_odf_c.resize(n_sh);
        prev_odf_z.resize(n_constraints);
        prev_odf_u.resize(n_constraints);

        {
          const Eigen::MatrixXd QTQ = params.Q.transpose() * params.Q;
          QTQ_free = QTQ.bottomRightCorner(nf, nf);
          QTQ_dc   = QTQ.block(1, 0, nf, 1);
          Q_free   = params.Q.rightCols(nf);
        }

        odf_K.resize(nf, nf);
        odf_c.resize(n_sh);
        odf_z.resize(n_constraints);
        odf_u.resize(n_constraints);
        odf_q.resize(n_constraints);
        odf_zold.resize(n_constraints);
        odf_rhs.resize(nf);
        odf_cfree.resize(nf);
        odf_admm_free.resize(nf);

        if (params.reg > 0.0 && n_shells > 1 && n_sh > 1)
        {
            const int n_reg_rows = (n_shells - 1) * (n_sh - 1);
            R_base.resize(n_reg_rows, n_valid_gauss);

            int row = 0;
            for (int j = 1; j < n_sh; ++j)
            {
                for (int s = 1; s < n_shells; ++s)
                {
                    for (int k = 0; k < n_valid_gauss; ++k)
                    {
                        const int g = params.valid_gaussian_indices[k];
                        R_base(row, k) = params.gaussians_rh[g](s, j);
                    }
                    ++row;
                }
            }

            RtR = R_base.transpose() * R_base;
        }
        else
        {
            RtR = Eigen::MatrixXd::Zero(n_valid_gauss, n_valid_gauss);
        }
      }
    };

  } // namespace

  static Eigen::VectorXd project_to_nonnegative(const Eigen::VectorXd& v)
  {
    return v.cwiseMax(0.0);
  }

  static Eigen::VectorXd init_fractions_isotropic(int n_valid_gauss, double total_mass, const Params& params)
  {
      Eigen::VectorXd fs_init =
          Eigen::VectorXd::Constant(
              n_valid_gauss,
              total_mass / std::max(1, n_valid_gauss));

      // Set anisotropic fractions to 0
      for (int k = 0; k < n_valid_gauss; ++k)
      {
          const double da = params.da[params.valid_gaussian_indices[k] / params.grid_size];
          const double dr = params.dr[params.valid_gaussian_indices[k] % params.grid_size];
          if (da > dr)
          {
              fs_init[k] = 0.0;
          }
      }
      return fs_init;
  }

  // Solves the constrained ODF subproblem via ADMM:
  //   minimize 0.5||H c - b||^2  s.t.  Q c >= 0,  c[0] fixed to the DC term.
  //
  // NOTE ON STRUCTURE: H is built (conceptually) so that column j only
  // touches rows [j*n_shells, (j+1)*n_shells) -- i.e. different columns
  // have disjoint row support. That means H^T H is exactly diagonal and
  // H^T b reduces to a per-column dot product, so H and b never need to
  // be formed at all: diagAtA(j) = sum_i rf(i,j)^2, Atb(j) = sum_i
  // rf(i,j)*S(i,j). This also means the correction term
  // "AtA.block(1,0,nf,1) * dc" from a dense normal-equation elimination
  // is structurally zero here and can be dropped.
  static void solve_odf_admm(
      const Eigen::MatrixXd& rf,
      const Eigen::MatrixXd& S,
      std::vector<double>& x0,
      const Params& params,
      int n_sh,
      Workspace& ws,
      int max_iter = 500)
  {

      const double dc = 1.0 / std::sqrt(4.0 * MR::Math::pi);
      const int nf = ws.nf;
      const int n_constraints = ws.n_constraints;

      const Eigen::VectorXd diagAtA = rf.colwise().squaredNorm().transpose();
      const Eigen::VectorXd Atb     = (rf.array() * S.array()).colwise().sum().transpose();

      const Eigen::VectorXd AtA_free_diag = diagAtA.tail(nf);
      const Eigen::VectorXd Atb_free      = Atb.tail(nf);

      Eigen::VectorXd& c = ws.odf_c;
      Eigen::VectorXd& z = ws.odf_z;
      Eigen::VectorXd& u = ws.odf_u;
      double rho = 1000.0;

      if (ws.odf_state_valid &&
          ws.prev_odf_c.size() == n_sh &&
          ws.prev_odf_z.size() == n_constraints &&
          ws.prev_odf_u.size() == n_constraints)
      {
          // Restore persistent ODF ADMM state from previous ALS iteration
          c = ws.prev_odf_c;
          z = ws.prev_odf_z;
          u = ws.prev_odf_u;
      }
      else
      {
          // Initialize ODF ADMM state for first ALS iteration or after reset
          rho = 1000.0;
          if (ws.prev_odf.size() == n_sh)
              c = ws.prev_odf;
          else
          {
              c.setZero();
              c[0] = dc;
          }
          z = (params.Q * c).cwiseMax(0.0);
          u.setZero();
      }

      const auto odf_quadratic_obj =
          [&diagAtA, &Atb](const Eigen::VectorXd& c) {
            return 0.5 * (diagAtA.array() * c.array().square()).sum() - Atb.dot(c);
          };


      Eigen::MatrixXd& K = ws.odf_K;
      K.noalias() = rho * ws.QTQ_free;
      K.diagonal() += AtA_free_diag;
      K.diagonal().array() += 1e-10;

      ws.odf_ldlt.compute(K);

      Eigen::VectorXd rho_qtq_dc_term = (rho * dc) * ws.QTQ_dc;

      Eigen::VectorXd& q = ws.odf_q;
      q.noalias() = params.Q * c;

      Eigen::VectorXd& rhs        = ws.odf_rhs;
      Eigen::VectorXd& c_free     = ws.odf_cfree;
      Eigen::VectorXd& admm_free  = ws.odf_admm_free;
      Eigen::VectorXd& z_old      = ws.odf_zold;

      double prev_obj = odf_quadratic_obj(c);
      int stagnation_count = 0;


      for (int k = 0; k < max_iter; ++k)
      {
          // c update
          admm_free.noalias() = ws.Q_free.transpose() * (z - u);
          rhs.noalias() = Atb_free + rho * admm_free - rho_qtq_dc_term;

          c_free = ws.odf_ldlt.solve(rhs);

          c[0] = dc;
          c.tail(nf) = c_free;

          // z update
          q.noalias() = params.Q * c;

          z_old = z;
          z = (q + u).cwiseMax(0.0);

          // dual update
          u += q - z;

          const double obj = odf_quadratic_obj(c);

          const double rel_obj =
              std::abs(obj - prev_obj)
              /
              std::max(std::abs(prev_obj), 1.0);

          if (rel_obj < 5e-4)
          {
              ++stagnation_count;
          }
          else
          {
              stagnation_count = 0;
          }

          prev_obj = obj;

          if (stagnation_count >= 5)
          {
              break;
          }
          
      }

      // Save persistent ODF ADMM state across ALS iterations
      ws.prev_odf_c = c;
      ws.prev_odf_z = z;
      ws.prev_odf_u = u;
      ws.odf_state_valid = true;

      ws.prev_odf = c;

      Eigen::Map<Eigen::VectorXd>(x0.data(), n_sh) = c;

      return;

  }

  static void rebuild_rf_from_fractions(
      Eigen::MatrixXd& rf,
      const std::vector<double>& x0,
      const Workspace& ws,
      const Params& params)
  {
      const int n_sh = ws.n_sh;

      rf.setZero();

      for (int k = 0; k < ws.n_valid_gauss; ++k)
      {
          const double f = x0[n_sh + k];

          rf.noalias() +=
              f *
              params.gaussians_rh[
                  params.valid_gaussian_indices[k]];
      }
  }

  // Solves: minimize 0.5*x'AtA*x - Atb'x  s.t. x >= 0
  // via ADMM on: minimize 0.5*x'AtA*x - Atb'x + I_{>=0}(z), x == z
  //
  // All working vectors/matrices (including the Cholesky factorization)
  // live in `ws` and are reused across calls
  static Eigen::VectorXd solve_fractions_admm(
      const Eigen::MatrixXd& AtA_in,
      const Eigen::VectorXd& Atb_in,
      const Eigen::VectorXd& warm,
      Workspace& ws,
      double rho          = 1.0,   // <=0 -> auto-select from trace(AtA)
      int    max_iter     = 1000,
      double rel_tol      = 1e-4,
      double relax        = 1.6,    // over-relaxation, 1.5-1.8 typical
      double lambda_ridge = 0.0,    // Tikhonov regularization weight, 0 = off
      bool   ridge_to_warm = true)  // regularize toward `warm` instead of 0
  {
      const int n = static_cast<int>(Atb_in.size());

      if (n == 0)
      {
          return Eigen::VectorXd();
      }

      // Fold ridge into the problem itself so the ADMM fixed point changes,
      // not just the per-iteration solve conditioning.
      Eigen::MatrixXd AtA = AtA_in;
      Eigen::VectorXd Atb = Atb_in;
      if (lambda_ridge > 0.0)
      {
          AtA.diagonal().array() += lambda_ridge;
          if (ridge_to_warm)
              Atb.noalias() += lambda_ridge * warm;
          // else: ridge toward zero, Atb unchanged
      }

      Eigen::VectorXd& x      = ws.frac_x;
      Eigen::VectorXd& z      = ws.frac_z;
      Eigen::VectorXd& u      = ws.frac_u;
      Eigen::VectorXd& x_hat  = ws.frac_xhat;
      Eigen::VectorXd& rhs    = ws.frac_rhs;
      Eigen::VectorXd& z_prev = ws.frac_zprev;

      if (ws.frac_state_valid &&
          ws.prev_frac_x.size() == n &&
          ws.prev_frac_z.size() == n &&
          ws.prev_frac_u.size() == n)
      {
          // Restore persistent fraction ADMM state from previous ALS iteration
          x = ws.prev_frac_x;
          z = ws.prev_frac_z;
          u = ws.prev_frac_u;
      }
      else
      {
          x = project_to_nonnegative(warm);
          z = x;
          u.setZero();
      }

      Eigen::MatrixXd& M = ws.frac_M;
      M = AtA;
      M.diagonal().array() += rho;
      ws.frac_llt.compute(M);
      if (ws.frac_llt.info() != Eigen::Success)
      {
          // Fall back to a slightly larger diagonal bump if AtA (even after
          // ridge) was borderline PSD and floating point pushed the
          // factorization negative.
          M.diagonal().array() += 1e-6 * std::max(1.0, AtA.trace() / n);
          ws.frac_llt.compute(M);
      }

      int stagnation_count = 0;

      int it = 0;
      for (; it < max_iter; ++it)
      {
          // x-update: (AtA + rho*I) x = Atb + rho*(z - u)
          rhs.noalias() = Atb + rho * (z - u);
          x = ws.frac_llt.solve(rhs);

          // over-relaxation
          x_hat.noalias() = relax * x + (1.0 - relax) * z;

          z_prev = z;

          // z-update: projection onto {z >= 0}
          z = (x_hat + u).cwiseMax(0.0);

          const double rel_z_change =
              (z - z_prev).norm()
              /
              std::max(z_prev.norm(), 1e-12);

          // dual update
          u += x_hat - z;

          // Stagnation criterion.
          // Only activate after ADMM had a chance to work.
          if (rel_z_change < rel_tol)
          {
              ++stagnation_count;
          }
          else
          {
              stagnation_count = 0;
          }
          
          if (stagnation_count >= 5)
          {
              ++it;
              break;
          }

      }
      
      // Save persistent fraction ADMM state across ALS iterations
      ws.prev_frac_x = x;
      ws.prev_frac_z = z;
      ws.prev_frac_u = u;
      ws.frac_state_valid = true;

      return z;
  }

  static void solve_fractions_fnnls(
      std::vector<double>& x0,
      Workspace& ws,
      const Params& params,
      int max_iter = 1000)
  {

      const int n_sh          = ws.n_sh;
      const int n_shells      = ws.n_shells;
      const int n_valid_gauss = ws.n_valid_gauss;

      Eigen::Map<const Eigen::VectorXd> odf(x0.data(), n_sh);

      const int M = n_shells * n_sh;
      Eigen::MatrixXd& H_data = ws.H_data;

      for (int k = 0; k < n_valid_gauss; ++k)
      {
          const int g = params.valid_gaussian_indices[k];
          const Eigen::MatrixXd& G = params.gaussians_rh[g];

          auto col = H_data.col(k);
          for (int j = 0; j < n_sh; ++j)
          {
            col.segment(j * n_shells, n_shells) = odf[j] * G.col(j);
          }
      }

      Eigen::Map<const Eigen::VectorXd> b_data(ws.S.data(), M);

      Eigen::MatrixXd& AtA = ws.AtA;
      Eigen::VectorXd& Atb = ws.Atb;

      // AtA is symmetric: compute only the lower triangle via rankUpdate
      // (roughly half the FLOPs of a dense H^T*H), then mirror it, since
      // downstream code (fraction_subproblem_objective, the ADMM x-update)
      // needs the full matrix.
      AtA.setZero();
      AtA.selfadjointView<Eigen::Lower>().rankUpdate(H_data.transpose());
      AtA.triangularView<Eigen::Upper>() = AtA.transpose();

      if (params.reg > 0.0)
      {
          AtA.noalias() += params.reg * ws.RtR;
      }
      Atb.noalias() = H_data.transpose() * b_data;

      Eigen::VectorXd warm;

      if (ws.prev_frac.size() == n_valid_gauss)
      {
          warm = project_to_nonnegative(ws.prev_frac);
      }
      else
      {
          warm = init_fractions_isotropic(n_valid_gauss, 1000.0, params);
      }

      const Eigen::VectorXd solve_result = solve_fractions_admm(AtA, Atb, warm, ws, .1, max_iter, 1e-4, 1.6);
      
      const Eigen::VectorXd& x = solve_result;

      ws.prev_frac = x;

      for (int k = 0; k < n_valid_gauss; ++k)
          x0[n_sh + k] = x[k];

      return;
  }

  static double rf_objective(
      const std::vector<double>& x,
      Workspace& ws,
      const Params& params)
  {
      const int n_sh = ws.n_sh;
      const int n_valid_gauss = ws.n_valid_gauss;
      const int n_shells = ws.n_shells;

      Eigen::Map<const Eigen::VectorXd> f(x.data() + n_sh, n_valid_gauss);
      Eigen::Map<const Eigen::VectorXd> odf(x.data(), n_sh);

      //
      // Reconstruct RF (reused buffer -- this is called on every
      // convergence check, so avoid a fresh allocation each time)
      //
      Eigen::MatrixXd& rf = ws.obj_rf;
      rf.setZero(n_shells, n_sh);

      for (int k = 0; k < n_valid_gauss; ++k)
      {
          const int g = params.valid_gaussian_indices[k];
          rf.noalias() += f[k] * params.gaussians_rh[g];
      }

      //
      // Predicted SH signal
      //
      Eigen::MatrixXd& pred = ws.obj_pred;
      pred = rf;

      for (int j = 0; j < n_sh; ++j)
          pred.col(j) *= odf[j];

      //
      // Data term
      //
      double obj = (pred - ws.S).squaredNorm();

      //
      // RF regularization
      //
      if (params.reg > 0.0)
      {
          obj += params.reg * (ws.R_base * f).squaredNorm();
      }

      return obj;
  }


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
    params.valid_gaussian_indices.clear();

    const double max_diffusivity = 3.3e-3;
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

    {
      int idx = 0;
      for (double da : params.da)
      {
        for (double dr : params.dr)
        {
          if (da >= dr)
            params.valid_gaussian_indices.push_back(idx);
          ++idx;
        }
      }
      params.n_valid_gauss = static_cast<int>(params.valid_gaussian_indices.size());
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

    // Reset persistent ADMM state when entering a completely new voxel fit
    // to prevent any voxel-to-voxel contamination.
    ws.frac_state_valid = false;

    ws.odf_state_valid = false;

    ws.prev_odf.resize(0);

    const int n_sh = n4l(params.lmax);
    const int n_gauss = params.grid_size * params.grid_size;
    const int n_valid_gauss = params.n_valid_gauss;
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

    // Scale S so that the largest coefficient at l=0 is 1000.0. This helps with numerical stability.
    const double base = 1000.0;
    const double scale = ws.S.block(0, 0, ws.S.rows(), 1).cwiseAbs().maxCoeff() / base;
    ws.S /= scale;

    Eigen::VectorXd fs_init = init_fractions_isotropic(n_valid_gauss, base, params);
    ws.prev_frac = fs_init;

    ws.rf.setZero();
    for (int k = 0; k < n_valid_gauss; ++k)
      ws.rf += fs_init[k] * params.gaussians_rh[params.valid_gaussian_indices[k]];

    // isotropic ODF init
    std::fill(ws.x0.begin(), ws.x0.begin() + n_sh, 0.0);
    ws.x0[0] = 1.0 / std::sqrt(4.0 * MR::Math::pi);

    for (int k = 0; k < n_valid_gauss; ++k)
    {
        ws.x0[n_sh + k] = fs_init[k];
    }

    if (ws.prev_odf.size() == n_sh)
    {
      for (int j = 0; j < n_sh; ++j)
        ws.x0[j] = ws.prev_odf[j];
      ws.x0[0] = 1.0 / std::sqrt(4.0 * MR::Math::pi);
    }

    const bool need_objectives = params.init_obj_fun || params.final_obj_fun;

    double init_obj = 0.0;
    double obj_before = 0.0;
    if (need_objectives)
    {
      init_obj = rf_objective(ws.x0, ws, params);
      obj_before = init_obj;
    }
    else
    {
      obj_before = rf_objective(ws.x0, ws, params);
    }


    const int max_als = 20;
    for (int outer = 0; outer < max_als; ++outer)
    {
      int admm_iter_limit = params.max_admm_iter;
      if (params.max_admm_iter_per_als > 0)
      {
          if (outer == 0)
          {
              admm_iter_limit = std::max(50, params.max_admm_iter_per_als * 2);
          }
          else
          {
              admm_iter_limit = params.max_admm_iter_per_als;
          }
      }

      {
        solve_odf_admm(
          ws.rf,
          ws.S,
          ws.x0,
          params,
          n_sh,
          ws,
          admm_iter_limit);
      }

      solve_fractions_fnnls(
          ws.x0,
          ws,
          params,
          admm_iter_limit);

      {
        rebuild_rf_from_fractions(
          ws.rf,
          ws.x0,
          ws,
          params);
      }
      

      double obj_after = 0.0;
      {
        obj_after =
          rf_objective(
              ws.x0,
              ws,
              params);
      }

      double rel_obj = std::abs(obj_after - obj_before) / std::max(std::abs(obj_before), 1.0);

      if (rel_obj < 1e-5 || outer == max_als - 1)
      {
          // If truncated ADMM mode was active, run one final polishing pass
          // with full max_admm_iter to ensure exact tolerance convergence.
          if (params.max_admm_iter_per_als > 0)
          {
              {
                solve_odf_admm(
                  ws.rf,
                  ws.S,
                  ws.x0,
                  params,
                  n_sh,
                  ws,
                  params.max_admm_iter);
              }

              solve_fractions_fnnls(
                  ws.x0,
                  ws,
                  params,
                  params.max_admm_iter);

              {
                rebuild_rf_from_fractions(
                  ws.rf,
                  ws.x0,
                  ws,
                  params);
              }
          }
          break;
      }

      obj_before = obj_after;

    }

    double obj_final = 0.0;
    if (need_objectives)
    {
      obj_final = rf_objective(ws.x0, ws, params);
    }

    Eigen::Map<Eigen::VectorXd> fs_valid(ws.x0.data() + n_sh, n_valid_gauss);

    Eigen::VectorXd fs_full = Eigen::VectorXd::Zero(n_gauss);

    // Normalize fs_valid to sum to 1.0
    const double sum_fs_valid = fs_valid.sum();
    if (sum_fs_valid > 0.0)
    {
        fs_valid /= sum_fs_valid;
    }

    for (int k = 0; k < n_valid_gauss; ++k)
    {
        fs_full[params.valid_gaussian_indices[k]] = fs_valid[k];
    }

    ws.kernel.setZero();

    for (int k = 0; k < n_valid_gauss; ++k)
    {
        const int g = params.valid_gaussian_indices[k];

        ws.kernel.noalias() += sum_fs_valid * fs_valid[k] * params.gaussians_rh[g];
    }

    Eigen::MatrixXd response = scale * rh2zh(ws.kernel, params.lmax);

    Eigen::MatrixXd response_rh = zh2rh(response, params.lmax);
    Eigen::VectorXd predicted = Eigen::VectorXd::Zero(static_cast<int>(dwi.size()));

    Eigen::Map<const Eigen::VectorXd> odf(ws.x0.data(), n_sh);

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
    result.fracs.assign(fs_full.data(), fs_full.data() + fs_full.size());
    const int n_orders = params.lmax / 2 + 1;

    result.response.assign(static_cast<size_t>(n_shells * n_orders), 0.0f);

    // Output order:
    // idx = shell * n_orders + order_index
    // order_index 0 -> l=0
    // order_index 1 -> l=2
    // order_index 2 -> l=4
    // ...
    for (int s = 0; s < n_shells; ++s)
    {
        for (int k = 0; k < n_orders; ++k)
        {
            const int idx = s * n_orders + k;
            result.response[static_cast<size_t>(idx)] = static_cast<float>(response(s, k));
        }
    }
    result.predicted_signal.assign(predicted.data(), predicted.data() + predicted.size());

    if (need_objectives)
    {
      result.f0 = init_obj;
      result.f1 = obj_final;
    }

    return result;
  }
}