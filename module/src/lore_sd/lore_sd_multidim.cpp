#include "lore_sd/lore_sd_multidim.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

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

        struct AtomIndex
        {
            int ida = 0;
            int idr = 0;
            int it2 = 0;
        };

        inline int total_atom_count(const Params &params)
        {
            return static_cast<int>(params.da.size() * params.dr.size() * params.t2.size());
        }

        inline AtomIndex decode_atom_index(int flat, const Params &params)
        {
            const int ndr = static_cast<int>(params.dr.size());
            const int nt2 = static_cast<int>(params.t2.size());

            if (ndr <= 0 || nt2 <= 0)
                throw std::invalid_argument("decode_atom_index requires non-empty dr and t2 grids");

            const int n_atoms = total_atom_count(params);
            if (flat < 0 || flat >= n_atoms)
                throw std::out_of_range("decode_atom_index flat index out of range");

            AtomIndex idx;
            idx.ida = flat / (ndr * nt2);
            const int rem = flat % (ndr * nt2);
            idx.idr = rem / nt2;
            idx.it2 = rem % nt2;
            return idx;
        }

        Eigen::MatrixXd expcoefs(const Eigen::VectorXd &a, const Eigen::VectorXd &beta, int lmax)
        {
            const int ncoeff = lmax / 2 + 1;
            const int nx = 10 * ncoeff + 1;
            Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(nx, -1.0, 1.0);
            x = x.tail(nx - 1);
            const Eigen::ArrayXd x2_centered = x.array().square() - 1.0 / 3.0;

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
                const Eigen::ArrayXd exponent = -a[i] * (beta[i] * x2_centered + 1.0 / 3.0);
                const Eigen::VectorXd y = exponent.exp().matrix();
                coefs.row(i) = qr.solve(y).transpose();
            }

            return coefs;
        }

        Eigen::MatrixXd zhgaussian(const std::vector<double> &bvals,
                                   const std::vector<double> &beta,
                                   const std::vector<double> &te,
                                   double Da,
                                   double Dr,
                                   double t2,
                                   int lmax)
        {
            const int n_shells = static_cast<int>(bvals.size());
            const int ncoeff = lmax / 2 + 1;

            const Eigen::VectorXd b_shell = Eigen::Map<const Eigen::VectorXd>(bvals.data(), n_shells);
            const Eigen::VectorXd a = b_shell.array() * (Da - Dr);
            const Eigen::VectorXd beta_shell = Eigen::Map<const Eigen::VectorXd>(beta.data(), n_shells);
            const Eigen::VectorXd te_shell = Eigen::Map<const Eigen::VectorXd>(te.data(), n_shells);
            const Eigen::MatrixXd expc = expcoefs(a, beta_shell, lmax);

            Eigen::VectorXd scale(ncoeff);
            for (int l = 0, col = 0; l <= lmax; l += 2, ++col)
            {
                scale[col] = std::sqrt(4.0 * MR::Math::pi / (2.0 * l + 1.0));
            }

            Eigen::MatrixXd gauss(n_shells, ncoeff);
            for (int i = 0; i < n_shells; ++i)
            {
                const double t2_decay = std::isfinite(t2) ? std::exp(-te_shell[i] / t2) : 1.0;
                for (int j = 0; j < ncoeff; ++j)
                {
                    gauss(i, j) = scale[j] * std::exp(-b_shell[i] * Dr) * expc(i, j) * t2_decay;
                }
            }

            const double b0_tol = 50.0;

            int ref_shell = -1;

            for (int s = 0; s < n_shells; ++s)
            {
                if (std::abs(b_shell[s]) <= b0_tol)
                {
                    ref_shell = s;
                    break;
                }
            }

            if (ref_shell >= 0)
            {

                const double norm = gauss(ref_shell, 0);

                if (std::abs(norm) > 1e-12)
                    gauss.array() /= norm;
            }

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
            const int n_gauss = total_atom_count(params);

            Eigen::MatrixXd gauss_zh(n_gauss, n_shells * ncoeff);
            gauss_zh.setZero();

            int idx = 0;
            for (double da : params.da)
            {
                for (double dr : params.dr)
                {
                    for (double t2 : params.t2)
                    {
                        const Eigen::MatrixXd zh =
                            zhgaussian(params.bvals, params.beta, params.te, da, dr, t2, params.lmax);
                        for (int s = 0; s < n_shells; ++s)
                        {
                            gauss_zh.block(idx, s * ncoeff, 1, ncoeff) = zh.row(s);
                        }
                        ++idx;
                    }
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

        void pack_measurements(const Eigen::VectorXd &dwi,
                               const Params &params,
                               Eigen::VectorXd &d)
        {
            int n_meas = 0;
            for (const auto &indices : params.shell_volumes)
                n_meas += static_cast<int>(indices.size());

            d.resize(n_meas);

            int offset = 0;
            for (const auto &indices : params.shell_volumes)
            {
                for (size_t i = 0; i < indices.size(); ++i)
                    d[offset + static_cast<int>(i)] = dwi[indices[i]];
                offset += static_cast<int>(indices.size());
            }
        }

        void predict_measurements_impl(const Eigen::MatrixXd &rf,
                                       const Eigen::VectorXd &odf,
                                       const Params &prediction_params,
                                       Eigen::VectorXd &predicted)
        {
            int n_meas = 0;
            for (const auto &indices : prediction_params.shell_volumes)
                n_meas += static_cast<int>(indices.size());

            predicted.resize(n_meas);
            predicted.setZero();

            int offset = 0;
            for (int s = 0; s < static_cast<int>(prediction_params.shell_volumes.size()); ++s)
            {
                const int ncoeff = prediction_params.shell_ncoeff[s];
                const int shell_size = static_cast<int>(prediction_params.shell_volumes[s].size());
                if (shell_size == 0)
                    continue;

                if (ncoeff > 0)
                {
                    Eigen::VectorXd shell_coeffs =
                        rf.row(s).head(ncoeff).transpose().cwiseProduct(odf.head(ncoeff));
                    predicted.segment(offset, shell_size).noalias() = prediction_params.shell_Q[s] * shell_coeffs;
                }

                offset += shell_size;
            }
        }

        struct Workspace
        {
            bool initialized = false;
            int n_sh = 0;
            int n_gauss = 0;
            int n_valid_gauss = 0;
            int n_shells = 0;
            int n_meas = 0;
            int nf = 0;            // n_sh - 1: "free" (l>0) ODF coefficients
            int n_constraints = 0; // rows of params.Q

            std::vector<double> x0;

            Eigen::MatrixXd S;
            Eigen::VectorXd d;
            Eigen::MatrixXd rf;
            Eigen::MatrixXd kernel;

            Eigen::MatrixXd R_base;
            Eigen::MatrixXd RtR;
            std::vector<int> shell_offsets;
            std::vector<Eigen::MatrixXd> shell_QtQ;
            std::vector<int> valid_atom_indices;
            std::vector<AtomIndex> valid_atoms;

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
            Eigen::MatrixXd QTQ_free; // nf x nf
            Eigen::VectorXd QTQ_dc;   // nf
            Eigen::MatrixXd Q_free;   // n_constraints x nf

            // -- ODF ADMM per-voxel working buffers --
            Eigen::MatrixXd odf_K;
            Eigen::LDLT<Eigen::MatrixXd> odf_ldlt;
            Eigen::VectorXd odf_c, odf_z, odf_u, odf_q, odf_zold, odf_rhs, odf_cfree, odf_admm_free;
            Eigen::MatrixXd odf_AtA;
            Eigen::VectorXd odf_Atb;

            // -- rf_objective scratch --
            Eigen::MatrixXd obj_rf;
            Eigen::VectorXd obj_pred_signal;

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
                const int n_gauss_local = total_atom_count(params);
                const int n_valid_gauss_local = params.n_valid_gauss;
                const int n_shells_local = static_cast<int>(params.bvals.size());
                int n_meas_local = 0;
                for (const auto &indices : params.shell_volumes)
                    n_meas_local += static_cast<int>(indices.size());

                if (initialized && n_sh == n_sh_local && n_gauss == n_gauss_local && n_valid_gauss == n_valid_gauss_local && n_shells == n_shells_local && n_meas == n_meas_local)
                    return;

                initialized = true;
                n_sh = n_sh_local;
                n_gauss = n_gauss_local;
                n_valid_gauss = n_valid_gauss_local;
                n_shells = n_shells_local;
                n_meas = n_meas_local;
                nf = n_sh - 1;
                n_constraints = static_cast<int>(params.Q.rows());

                x0.resize(n_sh + n_valid_gauss);
                S.resize(n_shells, n_sh);
                d.resize(n_meas);
                rf.resize(n_shells, n_sh);
                kernel.resize(n_shells, n_sh);
                shell_offsets.resize(n_shells);
                shell_QtQ.resize(n_shells);
                valid_atom_indices.resize(n_valid_gauss);
                valid_atoms.resize(n_valid_gauss);

                H_data.resize(n_meas, n_valid_gauss);
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
                    QTQ_dc = QTQ.block(1, 0, nf, 1);
                    Q_free = params.Q.rightCols(nf);
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
                odf_AtA.resize(n_sh, n_sh);
                odf_Atb.resize(n_sh);

                for (int k = 0; k < n_valid_gauss; ++k)
                {
                    valid_atom_indices[k] = params.valid_gaussian_indices[k];
                    valid_atoms[k] = decode_atom_index(valid_atom_indices[k], params);
                }

                if (params.reg > 0.0 && n_shells > 1 && n_sh > 1)
                {
                    const int n_reg_rows = (n_shells) * (n_sh - 1);
                    R_base.resize(n_reg_rows, n_valid_gauss);

                    int row = 0;
                    for (int j = 1; j < n_sh; ++j)
                    {
                        for (int s = 0; s < n_shells; ++s)
                        {
                            for (int k = 0; k < n_valid_gauss; ++k)
                            {
                                const int g = valid_atom_indices[k];
                                R_base(row, k) = params.gaussians_rh[g](s, j);
                            }
                            ++row;
                        }
                    }

                    RtR = R_base.transpose() * R_base;
                    RtR /= RtR.trace() / double(RtR.rows());
                
                }
                else
                {
                    RtR = Eigen::MatrixXd::Zero(n_valid_gauss, n_valid_gauss);
                }

                int offset = 0;
                for (int s = 0; s < n_shells; ++s)
                {
                    shell_offsets[s] = offset;
                    offset += static_cast<int>(params.shell_volumes[s].size());
                    if (params.shell_ncoeff[s] > 0)
                        shell_QtQ[s] = params.shell_Q[s].transpose() * params.shell_Q[s];
                    else
                        shell_QtQ[s].resize(0, 0);
                }

                obj_pred_signal.resize(n_meas);
            }
        };

    } // namespace

    static Eigen::VectorXd project_to_nonnegative(const Eigen::VectorXd &v)
    {
        return v.cwiseMax(0.0);
    }

    static Eigen::VectorXd init_fractions_isotropic(const Workspace &ws, double total_mass, const Params &params)
    {
        const int n_valid_gauss = ws.n_valid_gauss;
        Eigen::VectorXd fs_init =
            Eigen::VectorXd::Constant(
                n_valid_gauss,
                total_mass / std::max(1, n_valid_gauss));

        // Set anisotropic fractions to 0
        for (int k = 0; k < n_valid_gauss; ++k)
        {
            const AtomIndex atom = ws.valid_atoms[k];
            const double da = params.da[atom.ida];
            const double dr = params.dr[atom.idr];
            if (da > dr || da == 0.0)
            {
                fs_init[k] = 0.0;
            }
        }
        return fs_init;
    }

    // Solves the constrained ODF subproblem via ADMM:
    //   minimize 0.5||d - A(rf) c||^2  s.t.  Q c >= 0,  c[0] fixed to the DC term.
    //
    // The ODF remains parameterized in SH coefficients, but data fidelity is
    // evaluated in the native measurement space. The shell-wise sampling
    // matrices couple SH coefficients through Q_s^T Q_s, so the reduced normal
    // equations are dense rather than diagonal.
    static void solve_odf_admm(
        const Eigen::MatrixXd &rf,
        const Eigen::VectorXd &d,
        std::vector<double> &x0,
        const Params &params,
        int n_sh,
        Workspace &ws,
        int max_iter = 500)
    {

        const double dc = 1.0 / std::sqrt(4.0 * MR::Math::pi);
        const int nf = ws.nf;
        const int n_constraints = ws.n_constraints;

        Eigen::MatrixXd &AtA = ws.odf_AtA;
        Eigen::VectorXd &Atb = ws.odf_Atb;
        AtA.setZero();
        Atb.setZero();

        for (int s = 0; s < ws.n_shells; ++s)
        {
            const int ncoeff = params.shell_ncoeff[s];
            const int shell_size = params.shell_sizes[s];
            if (shell_size == 0 || ncoeff <= 0)
                continue;

            const int offset = ws.shell_offsets[s];
            const Eigen::VectorXd rf_shell = rf.row(s).head(ncoeff).transpose();
            const Eigen::MatrixXd weighted_qtq =
                rf_shell.asDiagonal() * ws.shell_QtQ[s] * rf_shell.asDiagonal();
            const Eigen::VectorXd weighted_qtd =
                rf_shell.asDiagonal() * (params.shell_Q[s].transpose() * d.segment(offset, shell_size));

            AtA.topLeftCorner(ncoeff, ncoeff).noalias() += weighted_qtq;
            Atb.head(ncoeff).noalias() += weighted_qtd;
        }

        const double N = double(ws.d.size());

        AtA /= N;
        Atb /= N;

        const Eigen::MatrixXd AtA_free = AtA.bottomRightCorner(nf, nf);
        const Eigen::VectorXd AtA_dc = AtA.block(1, 0, nf, 1);
        const Eigen::VectorXd Atb_free = Atb.tail(nf);

        Eigen::VectorXd &c = ws.odf_c;
        Eigen::VectorXd &z = ws.odf_z;
        Eigen::VectorXd &u = ws.odf_u;
        double rho = 10000.0;

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
            rho = 10000.0;
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
            [&AtA, &Atb](const Eigen::VectorXd &c)
        {
            return 0.5 * c.dot(AtA * c) - Atb.dot(c);
        };

        Eigen::MatrixXd &K = ws.odf_K;
        K.noalias() = AtA_free + rho * ws.QTQ_free;
        K.diagonal().array() += 1e-10;

        ws.odf_ldlt.compute(K);

        Eigen::VectorXd rho_qtq_dc_term = (rho * dc) * ws.QTQ_dc;

        Eigen::VectorXd &q = ws.odf_q;
        q.noalias() = params.Q * c;

        Eigen::VectorXd &rhs = ws.odf_rhs;
        Eigen::VectorXd &c_free = ws.odf_cfree;
        Eigen::VectorXd &admm_free = ws.odf_admm_free;
        Eigen::VectorXd &z_old = ws.odf_zold;

        double prev_obj = odf_quadratic_obj(c);
        int stagnation_count = 0;

        for (int k = 0; k < max_iter; ++k)
        {
            // c update
            admm_free.noalias() = ws.Q_free.transpose() * (z - u);
            rhs.noalias() = Atb_free - dc * AtA_dc + rho * admm_free - rho_qtq_dc_term;

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
                std::abs(obj - prev_obj) /
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
        Eigen::MatrixXd &rf,
        const std::vector<double> &x0,
        const Workspace &ws,
        const Params &params)
    {
        const int n_sh = ws.n_sh;

        rf.setZero();

        for (int k = 0; k < ws.n_valid_gauss; ++k)
        {
            const double f = x0[n_sh + k];

            rf.noalias() +=
                f *
                params.gaussians_rh[ws.valid_atom_indices[k]];
        }
    }

    // Solves: minimize 0.5*x'AtA*x - Atb'x  s.t. x >= 0
    // via ADMM on: minimize 0.5*x'AtA*x - Atb'x + I_{>=0}(z), x == z
    //
    // All working vectors/matrices (including the Cholesky factorization)
    // live in `ws` and are reused across calls
    static Eigen::VectorXd solve_fractions_admm(
        const Eigen::MatrixXd &AtA_in,
        const Eigen::VectorXd &Atb_in,
        const Eigen::VectorXd &warm,
        Workspace &ws,
        double rho = 10000.0, // <=0 -> auto-select from trace(AtA)
        int max_iter = 1000,
        double rel_tol = 1e-4,
        double relax = 1.6,        // over-relaxation, 1.5-1.8 typical
        double lambda_ridge = 0.0, // Tikhonov regularization weight, 0 = off
        bool ridge_to_warm = true) // regularize toward `warm` instead of 0
    {
        const int n = static_cast<int>(Atb_in.size());

        if (n == 0)
        {
            return Eigen::VectorXd();
        }

        const auto all_finite_vector = [](const Eigen::VectorXd &v)
        {
            return v.array().isFinite().all();
        };

        const auto all_finite_matrix = [](const Eigen::MatrixXd &m)
        {
            return m.array().isFinite().all();
        };

        if (!all_finite_matrix(AtA_in))
            throw std::runtime_error("fraction ADMM received non-finite AtA_in");
        if (!all_finite_vector(Atb_in))
            throw std::runtime_error("fraction ADMM received non-finite Atb_in");
        if (!all_finite_vector(warm))
            throw std::runtime_error("fraction ADMM received non-finite warm start; NaNs were present before ADMM iterations");

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

        Eigen::VectorXd &x = ws.frac_x;
        Eigen::VectorXd &z = ws.frac_z;
        Eigen::VectorXd &u = ws.frac_u;
        Eigen::VectorXd &x_hat = ws.frac_xhat;
        Eigen::VectorXd &rhs = ws.frac_rhs;
        Eigen::VectorXd &z_prev = ws.frac_zprev;

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

        if (!all_finite_matrix(AtA))
            throw std::runtime_error("fraction ADMM formed non-finite AtA after ridge update");
        if (!all_finite_vector(Atb))
            throw std::runtime_error("fraction ADMM formed non-finite Atb after ridge update");
        if (!all_finite_vector(x))
            throw std::runtime_error("fraction ADMM initialized non-finite x; NaNs were present before the first ADMM update");
        if (!all_finite_vector(z))
            throw std::runtime_error("fraction ADMM initialized non-finite z; NaNs were present before the first ADMM update");
        if (!all_finite_vector(u))
            throw std::runtime_error("fraction ADMM initialized non-finite u; NaNs were present before the first ADMM update");

        Eigen::MatrixXd &M = ws.frac_M;
        M = AtA;
        M.diagonal().array() += rho;
        if (!all_finite_matrix(M))
            throw std::runtime_error("fraction ADMM formed non-finite system matrix M before factorization");
        ws.frac_llt.compute(M);
        if (ws.frac_llt.info() != Eigen::Success)
        {
            // Fall back to a slightly larger diagonal bump if AtA (even after
            // ridge) was borderline PSD and floating point pushed the
            // factorization negative.
            M.diagonal().array() += 1e-6 * std::max(1.0, AtA.trace() / n);
            if (!all_finite_matrix(M))
                throw std::runtime_error("fraction ADMM formed non-finite fallback system matrix M");
            ws.frac_llt.compute(M);
        }

        int stagnation_count = 0;

        int it = 0;
        for (; it < max_iter; ++it)
        {
            // x-update: (AtA + rho*I) x = Atb + rho*(z - u)
            rhs.noalias() = Atb + rho * (z - u);
            if (!all_finite_vector(rhs))
                throw std::runtime_error("fraction ADMM produced non-finite rhs at iteration " + std::to_string(it));

            x = ws.frac_llt.solve(rhs);
            if (!all_finite_vector(x))
                throw std::runtime_error("fraction ADMM produced non-finite x at iteration " + std::to_string(it));

            // over-relaxation
            x_hat.noalias() = relax * x + (1.0 - relax) * z;
            if (!all_finite_vector(x_hat))
                throw std::runtime_error("fraction ADMM produced non-finite x_hat at iteration " + std::to_string(it));

            z_prev = z;

            // z-update: projection onto {z >= 0}
            z = (x_hat + u).cwiseMax(0.0);
            if (!all_finite_vector(z))
                throw std::runtime_error("fraction ADMM introduced non-finite z at iteration " + std::to_string(it));

            const double rel_z_change =
                (z - z_prev).norm() /
                std::max(z_prev.norm(), 1e-12);
            if (!std::isfinite(rel_z_change))
                throw std::runtime_error("fraction ADMM produced non-finite rel_z_change at iteration " + std::to_string(it));

            // dual update
            u += x_hat - z;
            if (!all_finite_vector(u))
                throw std::runtime_error("fraction ADMM introduced non-finite u at iteration " + std::to_string(it));

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
        std::vector<double> &x0,
        Workspace &ws,
        const Params &params,
        int max_iter = 1000)
    {

        const int n_sh = ws.n_sh;
        const int n_shells = ws.n_shells;
        const int n_valid_gauss = ws.n_valid_gauss;

        Eigen::Map<const Eigen::VectorXd> odf(x0.data(), n_sh);

        const int M = ws.n_meas;
        Eigen::MatrixXd &H_data = ws.H_data;

        for (int k = 0; k < n_valid_gauss; ++k)
        {
            const int g = ws.valid_atom_indices[k];
            H_data.col(k).setZero();

            for (int s = 0; s < n_shells; ++s)
            {
                const int ncoeff = params.shell_ncoeff[s];
                const int shell_size = params.shell_sizes[s];
                if (shell_size == 0)
                    continue;

                if (ncoeff > 0)
                {
                    Eigen::VectorXd shell_coeffs =
                        odf.head(ncoeff).cwiseProduct(
                            params.gaussians_rh[g].row(s).head(ncoeff).transpose());
                    H_data.block(ws.shell_offsets[s], k, shell_size, 1).noalias() =
                        params.shell_Q[s] * shell_coeffs;
                }
            }
        }

        const Eigen::VectorXd &b_data = ws.d;

        Eigen::MatrixXd &AtA = ws.AtA;
        Eigen::VectorXd &Atb = ws.Atb;

        AtA.setZero();
        AtA.selfadjointView<Eigen::Lower>().rankUpdate(H_data.transpose());
        AtA.triangularView<Eigen::Upper>() = AtA.transpose();

        Atb.noalias() = H_data.transpose() * b_data;

        const double N = double(b_data.size());

        AtA /= N;
        Atb /= N;

        if (params.reg > 0.0)
        {
            AtA.noalias() += params.reg * ws.RtR;
        }

        Eigen::VectorXd warm;

        if (ws.prev_frac.size() == n_valid_gauss)
        {
            warm = project_to_nonnegative(ws.prev_frac);
        }
        else
        {
            warm = init_fractions_isotropic(ws, 10000.0, params);
        }

        const Eigen::VectorXd solve_result = solve_fractions_admm(AtA, Atb, warm, ws, .001, max_iter, 1e-4, 1.6);

        const Eigen::VectorXd &x = solve_result;

        ws.prev_frac = x;

        for (int k = 0; k < n_valid_gauss; ++k)
            x0[n_sh + k] = x[k];

        return;
    }

    static double rf_objective(
        const std::vector<double> &x,
        Workspace &ws,
        const Params &params)
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
        Eigen::MatrixXd &rf = ws.obj_rf;
        rf.setZero(n_shells, n_sh);

        for (int k = 0; k < n_valid_gauss; ++k)
        {
            const int g = ws.valid_atom_indices[k];
            rf.noalias() += f[k] * params.gaussians_rh[g];
        }

        predict_measurements_impl(rf, odf, params, ws.obj_pred_signal);

       const double d_norm_sq = std::max(ws.d.squaredNorm(), 1e-12);
       const double N = double(ws.d.size());

       double obj = (ws.obj_pred_signal - ws.d).squaredNorm() / N;


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
    Params make_params_multidim(int lmax,
                                const int grid_size[3],
                                double reg,
                                const Eigen::MatrixXd &grad,
                                const Eigen::MatrixXd &eval_dirs,
                                const std::vector<double> &bvals,
                                const std::vector<std::vector<size_t>> &shell_volumes,
                                const std::vector<double> &beta,
                                const std::vector<double> &te)
    {
        Params params;
        params.lmax = lmax;
        for (int i = 0; i < 3; ++i)
            params.grid_size[i] = grid_size[i];
        params.reg = reg;
        params.bvals = bvals;
        const size_t n_shells_input = shell_volumes.size();
        const size_t n_volumes = static_cast<size_t>(grad.rows());

        if (beta.empty())
        {
            params.beta.assign(params.bvals.size(), 1.0);
        }
        else if (beta.size() == params.bvals.size())
        {
            params.beta = beta;
        }
        else if (beta.size() == n_volumes)
        {
            params.beta.assign(n_shells_input, 1.0);
            for (size_t s = 0; s < n_shells_input; ++s)
            {
                const auto &idx = shell_volumes[s];
                double ref = beta[idx.front()];
                for (size_t i = 0; i < idx.size(); ++i)
                {
                    if (std::abs(beta[idx[i]] - ref) > 1e-6)
                        throw std::runtime_error(
                            "shell contains multiple beta values");
                }
                params.beta[s] = ref;
            }
        }
        else
        {
            throw std::invalid_argument("beta must be empty, per-shell, or per-volume");
        }

        // No TE provided means T2 contribution is disabled, i.e. exp(-TE/T2) = 1.
        if (te.empty())
        {
            params.te.assign(params.bvals.size(), 0.0);
        }
        else if (te.size() == params.bvals.size())
        {
            params.te = te;
        }
        else if (te.size() == n_volumes)
        {
            params.te.assign(n_shells_input, 0.0);
            for (size_t s = 0; s < n_shells_input; ++s)
            {
                const auto &idx = shell_volumes[s];
                double ref = te[idx.front()];
                for (size_t i = 0; i < idx.size(); ++i)
                {
                    if (std::abs(te[idx[i]] - ref) > 1e-6)
                        throw std::runtime_error(
                            "shell contains multiple te values");
                }
                params.te[s] = ref;
            }
        }
        else
        {
            throw std::invalid_argument("te must be empty, per-shell, or per-volume");
        }

        if (params.beta.size() != params.bvals.size())
            throw std::invalid_argument("beta size must match bvals size");
        if (params.te.size() != params.bvals.size())
            throw std::invalid_argument("te size must match bvals size");

        params.da.resize(grid_size[0]);
        params.dr.resize(grid_size[1]);
        params.t2.resize(grid_size[2]);
        params.valid_gaussian_indices.clear();

        const double max_diffusivity = 3.3e-3;
        if (grid_size[0] <= 1)
        {
            params.da[0] = 0.0;
        }
        else
        {
            const double step = max_diffusivity / static_cast<double>(grid_size[0] - 1);
            for (int i = 0; i < grid_size[0]; ++i)
            {
                params.da[i] = step * static_cast<double>(i);
            }
        }

        if (grid_size[1] <= 1)
        {
            params.dr[0] = 0.0;
        }
        else
        {
            const double step = max_diffusivity / static_cast<double>(grid_size[1] - 1);
            for (int i = 0; i < grid_size[1]; ++i)
            {
                params.dr[i] = step * static_cast<double>(i);
            }
        }

        if (grid_size[2] <= 1)
        {
            params.t2[0] = std::numeric_limits<double>::infinity();
        }
        else
        {
            const double step = 140 / static_cast<double>(grid_size[2] - 2);
            for (int i = 0; i < grid_size[2] - 1; ++i)
            {
                params.t2[i] = 30.0 + step * static_cast<double>(i);
            }
            params.t2[grid_size[2] - 1] = 1500.0;
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
                if (bvals[s] > 10.0 && std::abs(params.beta[s]) > 1e-6)
                {
                    for (int l = 0; l <= lmax; l += 2)
                    {
                        nn = n4l(lmax);
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

        const int ndr = static_cast<int>(params.dr.size());
        const int nt2 = static_cast<int>(params.t2.size());
        const double eps = 1e-12;

        for (int ida = 0; ida < static_cast<int>(params.da.size()); ++ida)
        {
            const double da = params.da[ida];
            for (int idr = 0; idr < ndr; ++idr)
            {
                const double dr = params.dr[idr];
                for (int it2 = 0; it2 < nt2; ++it2)
                {
                    const double t2v = params.t2[it2];
                    bool valid = true;

                    if (da < dr || da == 0.0)
                        valid = false;

                    if (std::isfinite(t2v) &&
                        t2v >= 1500.0 &&
                        (da <= 3e-3 || dr <= 3e-3))
                    {
                        valid = false;
                    }

                    const int flat = (ida * ndr + idr) * nt2 + it2;

                    if (valid)
                        params.valid_gaussian_indices.push_back(flat);
                }
            }
        }

        params.n_valid_gauss =
            static_cast<int>(params.valid_gaussian_indices.size());

        params.gaussians_rh = compute_gaussians_rh(params);

        return params;
    }

    // Fit one voxel. This performs the per-voxel solve and packages outputs.
    Result fit_voxel_multidim(const Eigen::VectorXd &dwi, const Params &params)
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
        const int n_gauss = total_atom_count(params);
        const int n_valid_gauss = params.n_valid_gauss;
        const int n_shells = static_cast<int>(params.bvals.size());

        result.odf.assign(n_sh, 0.0f);
        result.fracs.assign(n_gauss, 0.0f);
        result.model_weights.assign(n_gauss, 0.0f);
        result.response.assign(n_shells * (params.lmax / 2 + 1), 0.0f);
        result.predicted_signal.assign(static_cast<size_t>(dwi.size()), 0.0f);

        if (dwi.size() == 0)
            return result;

        compute_signal_sh(dwi, params, ws.S, ws.shell_signal);
        if (ws.S(0, 0) == 0.0)
            return result;

        // Scale S so that the largest coefficient at l=0 is 10000.0. This helps with numerical stability.
        const double base = 10000.0;
        const double scale = ws.S.block(0, 0, ws.S.rows(), 1).cwiseAbs().maxCoeff() / base;
        ws.S /= scale;
        pack_measurements(dwi, params, ws.d);
        ws.d /= scale;

        Eigen::VectorXd fs_init = init_fractions_isotropic(ws, 1, params);
        ws.prev_frac = fs_init;

        ws.rf.setZero();
        for (int k = 0; k < n_valid_gauss; ++k)
            ws.rf += fs_init[k] * params.gaussians_rh[ws.valid_atom_indices[k]];

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

        // Reconstruct the signal and find the appropriate scaling of the fractions to match signal intensity
        Eigen::VectorXd init_pred = Eigen::VectorXd::Zero(static_cast<int>(dwi.size()));

        predict_measurements_impl(ws.rf, Eigen::Map<const Eigen::VectorXd>(ws.x0.data(), n_sh), params, init_pred);
        // First find the measurements corresponding to b=0
        std::vector<int> b0_indices;
        for (size_t i = 0; i < dwi.size(); ++i)
        {
            if (params.bvals[i] == 0)
                b0_indices.push_back(static_cast<int>(i));
        }
        Eigen::VectorXd b0_signal(b0_indices.size());
        for (size_t i = 0; i < b0_indices.size(); ++i)
        {
            b0_signal[i] = dwi[b0_indices[i]];
        }
        Eigen::VectorXd pred_b0_signal(b0_indices.size());
        for (size_t i = 0; i < b0_indices.size(); ++i)
        {
            pred_b0_signal[i] = init_pred[b0_indices[i]];
        }
        
        // Compute the scaling factor for the fractions based on the b=0 signal
        double scale_factor = 1.0;
        if (b0_signal.size() > 0)
        {
            scale_factor = b0_signal.mean() / pred_b0_signal.mean();
            for (int k = 0; k < n_valid_gauss; ++k)
            {
                ws.x0[n_sh + k] *= scale_factor;
            }
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
                    ws.d,
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
                            ws.d,
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

        const Eigen::VectorXd fs_valid_raw = fs_valid;
        Eigen::VectorXd fs_full = Eigen::VectorXd::Zero(n_gauss);
        Eigen::VectorXd model_weights_full = Eigen::VectorXd::Zero(n_gauss);

        // Normalize fs_valid to sum to 1.0
        const double sum_fs_valid = fs_valid.sum();
        if (sum_fs_valid > 0.0)
        {
            fs_valid /= sum_fs_valid;
        }

        for (int k = 0; k < n_valid_gauss; ++k)
        {
            fs_full[ws.valid_atom_indices[k]] = fs_valid[k];
            model_weights_full[ws.valid_atom_indices[k]] = scale * fs_valid_raw[k];
        }

        ws.kernel.setZero();

        for (int k = 0; k < n_valid_gauss; ++k)
        {
            const int g = ws.valid_atom_indices[k];

            ws.kernel.noalias() += sum_fs_valid * fs_valid[k] * params.gaussians_rh[g];
        }

        Eigen::MatrixXd response = scale * rh2zh(ws.kernel, params.lmax);

        Eigen::MatrixXd response_rh = zh2rh(response, params.lmax);
        Eigen::VectorXd predicted_stacked;
        Eigen::VectorXd predicted = Eigen::VectorXd::Zero(static_cast<int>(dwi.size()));

        Eigen::Map<const Eigen::VectorXd> odf(ws.x0.data(), n_sh);
        predict_measurements_impl(response_rh, odf, params, predicted_stacked);
        for (int s = 0; s < n_shells; ++s)
        {
            const auto &indices = params.shell_volumes[s];
            const int offset = ws.shell_offsets[s];
            for (size_t i = 0; i < indices.size(); ++i)
                predicted[static_cast<int>(indices[i])] = predicted_stacked[offset + static_cast<int>(i)];
        }

        result.odf.assign(odf.data(), odf.data() + odf.size());
        result.fracs.assign(fs_full.data(), fs_full.data() + fs_full.size());
        result.model_weights.assign(model_weights_full.data(), model_weights_full.data() + model_weights_full.size());
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

    void predict_measurements(const Eigen::MatrixXd &rf,
                              const Eigen::VectorXd &odf,
                              const Params &prediction_params,
                              Eigen::VectorXd &predicted)
    {
        predict_measurements_impl(rf, odf, prediction_params, predicted);
    }

    void predict_from_fit_result(const Result &fit_result,
                                 const Params &prediction_params,
                                 Eigen::VectorXd &predicted)
    {
        const int n_shells = static_cast<int>(prediction_params.bvals.size());
        const int n_sh = n4l(prediction_params.lmax);
        const int n_gauss = total_atom_count(prediction_params);

        if (static_cast<int>(fit_result.odf.size()) != n_sh)
            throw std::invalid_argument("fit ODF size is incompatible with prediction params");
        if (static_cast<int>(fit_result.model_weights.size()) != n_gauss)
            throw std::invalid_argument("fit model weights size is incompatible with prediction params");

        Eigen::MatrixXd rf_rh = Eigen::MatrixXd::Zero(n_shells, n_sh);
        for (int g = 0; g < n_gauss; ++g)
        {
            const double weight = fit_result.model_weights[static_cast<size_t>(g)];
            if (weight == 0.0)
                continue;
            rf_rh.noalias() += weight * prediction_params.gaussians_rh[static_cast<size_t>(g)];
        }

        const Eigen::VectorXd odf =
        Eigen::Map<const Eigen::VectorXf>(
            fit_result.odf.data(),
            n_sh).cast<double>();
        predict_measurements_impl(rf_rh, odf, prediction_params, predicted);
    }
}