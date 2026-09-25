/* Copyright (c) 2026.
 *
 * External MRtrix3 module for LoRE-SD.
 */

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <memory>


#include "command.h"
#include "header.h"
#include "image.h"
#include "algo/threaded_loop.h"
#include "dwi/gradient.h"
#include "dwi/directions/predefined.h"
#include "math/SH.h"
#include "metadata/phase_encoding.h"
#include "stride.h"

#include "lore_sd/lore_sd_multidim.h"

using namespace MR;
using namespace App;

namespace
{
    struct ShellInfo
    {
        double b = 0.0;
        double beta = 1.0;
        double te = 0.0;
        std::vector<size_t> volumes;
    };

    bool close_enough(double a, double b, double tol)
    {
        return std::abs(a - b) <= tol;
    }

    std::vector<double> bvals_per_volume(const Eigen::MatrixXd& grad)
    {
        std::vector<double> bvals(static_cast<size_t>(grad.rows()), 0.0);
        for (size_t i = 0; i < bvals.size(); ++i)
            bvals[i] = grad(static_cast<Eigen::Index>(i), 3);
        return bvals;
    }

    std::vector< std::vector<size_t> > group_by_b(const std::vector<double>& bvals, double b_tol)
    {
        std::vector< std::vector<size_t> > groups;
        for (size_t v = 0; v < bvals.size(); ++v)
        {
            bool placed = false;
            for (auto& g : groups)
            {
                if (g.empty())
                    continue;
                const double bref = bvals[g.front()];
                if (close_enough(bvals[v], bref, b_tol))
                {
                    g.push_back(v);
                    placed = true;
                    break;
                }
            }
            if (!placed)
                groups.push_back(std::vector<size_t>{v});
        }
        return groups;
    }

    std::vector<double> expand_protocol_values(
        const std::vector<double>& provided,
        const std::vector< std::vector<size_t> >& b_groups,
        size_t n_volumes,
        double default_value,
        const std::string& name)
    {
        if (provided.empty())
            return std::vector<double>(n_volumes, default_value);

        if (provided.size() == n_volumes)
            return provided;

        if (provided.size() == b_groups.size())
        {
            std::vector<double> expanded(n_volumes, default_value);
            for (size_t s = 0; s < b_groups.size(); ++s)
            {
                for (size_t idx : b_groups[s])
                    expanded[idx] = provided[s];
            }
            return expanded;
        }

        throw Exception(name + " input must contain either one value per volume or one value per b-shell");
    }

    std::vector<ShellInfo> group_shells_b_beta_te(const std::vector<double>& bvals,
                                                  const std::vector<double>& beta,
                                                  const std::vector<double>& te,
                                                  double b_tol,
                                                  double beta_tol,
                                                  double te_tol)
    {
        std::vector<ShellInfo> shells;
        for (size_t v = 0; v < bvals.size(); ++v)
        {
            bool placed = false;
            for (auto& sh : shells)
            {
                if (close_enough(bvals[v], sh.b, b_tol) &&
                    close_enough(beta[v], sh.beta, beta_tol) &&
                    close_enough(te[v], sh.te, te_tol))
                {
                    sh.volumes.push_back(v);
                    placed = true;
                    break;
                }
            }

            if (!placed)
            {
                ShellInfo sh;
                sh.b = bvals[v];
                sh.beta = beta[v];
                sh.te = te[v];
                sh.volumes.push_back(v);
                shells.push_back(sh);
            }
        }
        return shells;
    }
}

static std::vector<double> load_numeric_list_file(const std::string& path)
{
    std::ifstream in(path.c_str());
    if (!in)
        throw Exception("cannot open numeric input file: " + path);

    std::vector<double> values;
    std::string line;
    while (std::getline(in, line))
    {
        const size_t comment = line.find('#');
        if (comment != std::string::npos)
            line = line.substr(0, comment);

        std::istringstream iss(line);
        double v;
        while (iss >> v)
            values.push_back(v);
    }

    if (values.empty())
        throw Exception("numeric input file is empty: " + path);

    return values;
}

static std::vector< std::vector<double> > load_numeric_matrix_file(const std::string& path)
{
    std::ifstream in(path.c_str());
    if (!in)
        throw Exception("cannot open numeric input file: " + path);

    std::vector< std::vector<double> > rows;
    std::string line;
    while (std::getline(in, line))
    {
        const size_t comment = line.find('#');
        if (comment != std::string::npos)
            line = line.substr(0, comment);

        std::istringstream iss(line);
        std::vector<double> row;
        double v;
        while (iss >> v)
            row.push_back(v);

        if (!row.empty())
            rows.push_back(row);
    }

    if (rows.empty())
        throw Exception("numeric input file is empty: " + path);

    return rows;
}

static Eigen::MatrixXd load_recon_grad_from_fsl(const std::string& bvec_path,
                                                const std::string& bval_path)
{
    const auto bvec_rows = load_numeric_matrix_file(bvec_path);
    const auto bvals = load_numeric_list_file(bval_path);
    const size_t n_volumes = bvals.size();

    Eigen::MatrixXd grad(static_cast<Eigen::Index>(n_volumes), 4);

    if (bvec_rows.size() == 3)
    {
        const size_t n_cols = bvec_rows[0].size();
        if (bvec_rows[1].size() != n_cols || bvec_rows[2].size() != n_cols)
            throw Exception("recon_bvec rows must have the same number of columns");
        if (n_cols != n_volumes)
            throw Exception("recon_bvec and recon_bval must describe the same number of volumes");

        for (size_t v = 0; v < n_volumes; ++v)
        {
            grad(static_cast<Eigen::Index>(v), 0) = bvec_rows[0][v];
            grad(static_cast<Eigen::Index>(v), 1) = bvec_rows[1][v];
            grad(static_cast<Eigen::Index>(v), 2) = bvec_rows[2][v];
            grad(static_cast<Eigen::Index>(v), 3) = bvals[v];
        }
        return grad;
    }

    if (bvec_rows.size() == n_volumes)
    {
        for (size_t v = 0; v < n_volumes; ++v)
        {
            if (bvec_rows[v].size() != 3)
                throw Exception("recon_bvec must be either a 3xN FSL file or an Nx3 table");
            grad(static_cast<Eigen::Index>(v), 0) = bvec_rows[v][0];
            grad(static_cast<Eigen::Index>(v), 1) = bvec_rows[v][1];
            grad(static_cast<Eigen::Index>(v), 2) = bvec_rows[v][2];
            grad(static_cast<Eigen::Index>(v), 3) = bvals[v];
        }
        return grad;
    }

    throw Exception("recon_bvec must be either a 3xN FSL file or an Nx3 table matching recon_bval");
}

void usage()
{
    AUTHOR = "Siebe Leysen";

    SYNOPSIS = "Estimate fibre orientation distributions using LoRE-SD";

    DESCRIPTION
    +"This is a standalone LoRE-SD implementation built as an MRtrix3 external module.";

    ARGUMENTS
    +Argument("dwi", "the input diffusion-weighted image").type_image_in() + Argument("odf", "output ODF image").type_image_out() + Argument("fracs", "output LoRE-SD fractions image").type_image_out() + Argument("response", "output LoRE-SD response image").type_image_out();

    OPTIONS
    +DWI::GradImportOptions()
    + Option("mask", "only perform computation within the specified binary brain mask image")
     + Argument("image").type_image_in()
    + Option("lmax", "maximum spherical harmonic order (default: 8)")
     + Argument("order").type_integer(8)
    + Option("grid_da", "Da grid size")
     + Argument("size").type_integer(10)
    + Option("grid_dr", "Dr grid size")
      + Argument("size").type_integer(10)
    + Option("grid_t2", "T2 grid size")
     + Argument("size").type_integer(1)
    + Option("reg", "regularisation parameter (default: 4e-5)")
     + Argument("value").type_float(4e-5)
    + Option("beta", "path to beta values file (one value per shell or per volume)")
     + Argument("file").type_file_in()
    + Option("te", "path to TE values file (one value per shell or per volume)")
     + Argument("file").type_file_in()
    + Option("init_obj_fun", "write initial objective function per voxel to an output image")
     + Argument("image").type_image_out()
    + Option("final_obj_fun", "write final objective function per voxel to an output image")
     + Argument("image").type_image_out()
    + Option("predicted_signal", "write the DWI signal predicted by LoRE-SD to an output image")
        + Argument("image").type_image_out()
    + Option("recon_grad", "enable reconstruction onto a different acquisition scheme")
    + Option("recon_bvec", "path to reconstruction bvec file (required when recon_grad is used)")
        + Argument("file").type_file_in()
    + Option("recon_bval", "path to reconstruction bval file (required when recon_grad is used)")
        + Argument("file").type_file_in()
    + Option("recon_beta", "path to reconstruction beta values file (optional, one value per shell or per volume)")
        + Argument("file").type_file_in()
    + Option("recon_te", "path to reconstruction TE values file (optional, one value per shell or per volume)")
        + Argument("file").type_file_in()
    + Option("dwi_recon", "write reconstructed DWI signal on the reconstruction scheme (required when recon_grad is used)")
        + Argument("image").type_image_out()
    + Stride::Options;
}

class LoRESD_Processor
{
    MEMALIGN(LoRESD_Processor)
public:
    LoRESD_Processor(const LoreSD::Params &params,
                     const LoreSD::Params *recon_params,
                     Image<bool> &mask,
                     Image<float> &odf,
                     Image<float> &fracs,
                     Image<float> &response,
                     Image<float> &dwi_recon,
                     Image<float> &predicted_signal,
                     Image<float> &init_obj_fun,
                     Image<float> &final_obj_fun) : params(params),
                                                                have_recon_params(recon_params != nullptr),
                                                                recon_params(recon_params ? *recon_params : LoreSD::Params()),
                                                                mask(mask),
                                                                odf_image(odf),
                                                                fracs_image(fracs),
                                                                response_image(response),
                                                                dwi_recon_image(dwi_recon),
                                                                predicted_signal_image(predicted_signal),
                                                                init_obj_fun_image(init_obj_fun),
                                                                final_obj_fun_image(final_obj_fun),
                                                                dwi_data(0)
    {
        n_odf = MR::Math::SH::NforL(params.lmax);
        n_fracs = params.da.size() * params.dr.size() * params.t2.size();
        n_response = static_cast<size_t>(params.bvals.size()) * static_cast<size_t>(params.lmax / 2 + 1);
        n_predicted_signal = 0;
        n_dwi_recon = have_recon_params ? total_measurements(this->recon_params.shell_volumes) : 0;
        if (have_recon_params)
        {
            recon_shell_offsets.resize(this->recon_params.shell_volumes.size());
            int offset = 0;
            for (size_t s = 0; s < this->recon_params.shell_volumes.size(); ++s)
            {
                recon_shell_offsets[s] = offset;
                offset += static_cast<int>(this->recon_params.shell_volumes[s].size());
            }
        }
    }

    void operator()(Image<float> &dwi)
    {
        n_predicted_signal = static_cast<size_t>(dwi.size(3));

        if (mask.valid())
        {
            assign_pos_of(dwi, 0, 3).to(mask);
            if (!mask.value())
            {
                write_zero_outputs(dwi);
                return;
            }
        }

        if (!dwi_data.size())
            dwi_data.resize(dwi.size(3));

        for (size_t n = 0; n < static_cast<size_t>(dwi.size(3)); ++n)
        {
            dwi.index(3) = n;
            const float val = dwi.value();
            dwi_data[n] = std::isfinite(val) ? std::max(0.0f, val) : 0.0f;
        }

        auto result = LoreSD::fit_voxel_multidim(dwi_data, params);

        write_vector(dwi, odf_image, result.odf, n_odf);
        write_vector(dwi, fracs_image, result.fracs, n_fracs);
        write_vector(dwi, response_image, result.response, n_response);
        if (have_recon_params)
        {
            Eigen::VectorXd recon_stacked;
            LoreSD::predict_from_fit_result(result, recon_params, recon_stacked);

            std::vector<float> recon_by_volume(n_dwi_recon, 0.0f);
            for (size_t s = 0; s < recon_params.shell_volumes.size(); ++s)
            {
                const auto &indices = recon_params.shell_volumes[s];
                const int offset = recon_shell_offsets[s];
                for (size_t i = 0; i < indices.size(); ++i)
                {
                    recon_by_volume[indices[i]] =
                        static_cast<float>(recon_stacked[offset + static_cast<int>(i)]);
                }
            }
            write_vector(dwi, dwi_recon_image, recon_by_volume, n_dwi_recon);
        }
        write_vector(dwi, predicted_signal_image, result.predicted_signal, n_predicted_signal);
        write_scalar(dwi, init_obj_fun_image, result.f0);
        write_scalar(dwi, final_obj_fun_image, result.f1);

        (void)result; // no CSV profiling in public release
    }

private:
    static size_t total_measurements(const std::vector< std::vector<size_t> >& shell_volumes)
    {
        size_t n = 0;
        for (const auto& shell : shell_volumes)
            n += shell.size();
        return n;
    }

    LoreSD::Params params;
    bool have_recon_params = false;
    LoreSD::Params recon_params;
    Image<bool> mask;
    Image<float> odf_image;
    Image<float> fracs_image;
    Image<float> response_image;
    Image<float> dwi_recon_image;
    Image<float> predicted_signal_image;
    Image<float> init_obj_fun_image;
    Image<float> final_obj_fun_image;
    Eigen::VectorXd dwi_data;
    size_t n_odf = 0;
    size_t n_fracs = 0;
    size_t n_response = 0;
    size_t n_dwi_recon = 0;
    size_t n_predicted_signal = 0;
    std::vector<int> recon_shell_offsets;

    void write_zero_outputs(Image<float> &dwi)
    {
        static const std::vector<float> empty;
        write_vector(dwi, odf_image, empty, n_odf);
        write_vector(dwi, fracs_image, empty, n_fracs);
        write_vector(dwi, response_image, empty, n_response);
        write_vector(dwi, dwi_recon_image, empty, n_dwi_recon);
        write_vector(dwi, predicted_signal_image, empty, n_predicted_signal);
        write_scalar(dwi, init_obj_fun_image, 0.0f);
        write_scalar(dwi, final_obj_fun_image, 0.0f);
    }

    void write_scalar(Image<float> &dwi, Image<float> &image, float value)
    {
        if (!image.valid())
            return;
        assign_pos_of(dwi, 0, 3).to(image);
        image.value() = value;
    }

    void write_vector(Image<float> &dwi, Image<float> &image, const std::vector<float> &data, size_t expected)
    {
        if (!image.valid())
            return;

        assign_pos_of(dwi, 0, 3).to(image);
        size_t idx = 0;

        if (image.ndim() == 4)
        {
            for (size_t c = 0; c < expected; ++c)
            {
                image.index(3) = c;
                image.value() = (idx < data.size()) ? data[idx++] : 0.0f;
            }
            return;
        }

        if (image.ndim() == 5)
        {
            const size_t dim3 = static_cast<size_t>(image.size(3));  // ad
            const size_t dim4 = static_cast<size_t>(image.size(4));  // rd
            // Write with dim3 (ad) as outer loop, dim4 (rd) as inner
            // This matches C++ generation order: for da: for dr:
            for (size_t i = 0; i < dim3; ++i)
            {
                image.index(3) = i;
                for (size_t j = 0; j < dim4; ++j)
                {
                    image.index(4) = j;
                    image.value() = (idx < data.size()) ? data[idx++] : 0.0f;
                }
            }
            return;
        }

        if (image.ndim() == 6)
        {
            const size_t dim3 = static_cast<size_t>(image.size(3));  // ad
            const size_t dim4 = static_cast<size_t>(image.size(4));  // rd
            const size_t dim5 = static_cast<size_t>(image.size(5));  // t2
            for (size_t i = 0; i < dim3; ++i)
            {
                image.index(3) = i;
                for (size_t j = 0; j < dim4; ++j)
                {
                    image.index(4) = j;
                    for (size_t k = 0; k < dim5; ++k)
                    {
                        image.index(5) = k;
                        image.value() = (idx < data.size()) ? data[idx++] : 0.0f;
                    }
                }
            }
            return;
        }

        for (size_t c = 0; c < expected; ++c)
        {
            image.index(3) = c;
            image.value() = (idx < data.size()) ? data[idx++] : 0.0f;
        }
    }
};

static Eigen::MatrixXd load_default_eval_dirs()
{
    return DWI::Directions::tesselation_129();
}

void run()
{
    auto header_in = Header::open(argument[0]);
    auto grad = DWI::get_DW_scheme(header_in);
    if (grad.cols() < 4)
        throw Exception("DW scheme must have at least 4 columns (gx gy gz b)");

    const Eigen::MatrixXd eval_dirs = load_default_eval_dirs();

    auto mask = Image<bool>();
    auto opt = get_options("mask");
    if (opt.size())
    {
        mask = Header::open(opt[0][0]).get_image<bool>();
        check_dimensions(header_in, mask, 0, 3);
    }

    int lmax = 8;
    opt = get_options("lmax");
    if (opt.size())
    {
        lmax = to<int>(opt[0][0]);
        if (lmax % 2)
            throw Exception("lmax must be an even number");
    }

    int grid_size[3] = {10,10,1};

    opt = get_options("grid_da");
    if (opt.size())
        grid_size[0] = to<int>(opt[0][0]);

    opt = get_options("grid_dr");
    if (opt.size())
        grid_size[1] = to<int>(opt[0][0]);

    opt = get_options("grid_t2");
    if (opt.size())
        grid_size[2] = to<int>(opt[0][0]);

    double reg = 5e-5;
    opt = get_options("reg");
    if (opt.size())
        reg = to<double>(opt[0][0]);

    bool have_init_obj_fun = false;
    std::string init_obj_fun_path;
    opt = get_options("init_obj_fun");
    if (opt.size())
    {
        have_init_obj_fun = true;
        init_obj_fun_path = std::string(opt[0][0]);
    }

    bool have_final_obj_fun = false;
    std::string final_obj_fun_path;
    opt = get_options("final_obj_fun");
    if (opt.size())
    {
        have_final_obj_fun = true;
        final_obj_fun_path = std::string(opt[0][0]);
    }

    bool have_predicted_signal = false;
    std::string predicted_signal_path;
    opt = get_options("predicted_signal");
    if (opt.size())
    {
        have_predicted_signal = true;
        predicted_signal_path = std::string(opt[0][0]);
    }

    std::vector<double> beta_values;
    opt = get_options("beta");
    if (opt.size())
        beta_values = load_numeric_list_file(std::string(opt[0][0]));

    std::vector<double> te_values;
    opt = get_options("te");
    if (opt.size())
        te_values = load_numeric_list_file(std::string(opt[0][0]));

    const bool have_recon_grad = get_options("recon_grad").size();
    std::string recon_bvec_path;
    std::string recon_bval_path;
    std::string dwi_recon_path;
    std::vector<double> recon_beta_values;
    std::vector<double> recon_te_values;

    if (have_recon_grad)
    {
        auto recon_bvec_opt = get_options("recon_bvec");
        auto recon_bval_opt = get_options("recon_bval");
        auto dwi_recon_opt = get_options("dwi_recon");

        if (!recon_bvec_opt.size())
            throw Exception("recon_bvec is required when recon_grad is used");
        if (!recon_bval_opt.size())
            throw Exception("recon_bval is required when recon_grad is used");
        if (!dwi_recon_opt.size())
            throw Exception("dwi_recon is required when recon_grad is used");

        recon_bvec_path = std::string(recon_bvec_opt[0][0]);
        recon_bval_path = std::string(recon_bval_opt[0][0]);
        dwi_recon_path = std::string(dwi_recon_opt[0][0]);

        opt = get_options("recon_beta");
        if (opt.size())
            recon_beta_values = load_numeric_list_file(std::string(opt[0][0]));

        opt = get_options("recon_te");
        if (opt.size())
            recon_te_values = load_numeric_list_file(std::string(opt[0][0]));
    }

    std::vector<double> bvals;
    std::vector<double> shell_beta;
    std::vector<double> shell_te;
    std::vector<std::vector<size_t>> shell_volumes;

    const auto b_per_vol = bvals_per_volume(grad);
    const auto b_groups = group_by_b(b_per_vol, 50.0);
    const auto beta_per_vol = expand_protocol_values(beta_values, b_groups, b_per_vol.size(), 1.0, "beta");
    const auto te_per_vol = expand_protocol_values(te_values, b_groups, b_per_vol.size(), 0.0, "te");

    const auto shells = group_shells_b_beta_te(b_per_vol, beta_per_vol, te_per_vol, 50.0, 1e-2, 1e-2);

    bvals.reserve(shells.size());
    shell_beta.reserve(shells.size());
    shell_te.reserve(shells.size());
    shell_volumes.reserve(shells.size());
    for (const auto& sh : shells)
    {
        bvals.push_back(sh.b);
        shell_beta.push_back(sh.beta);
        shell_te.push_back(sh.te);
        shell_volumes.push_back(sh.volumes);
    }

    CONSOLE("DWI gradient table: " + std::to_string(grad.rows()) + " volumes, " +
            std::to_string(shells.size()) + " shells: ");
    for (size_t i = 0; i < bvals.size(); ++i)
        CONSOLE("  Shell " + std::to_string(i) + ": b=" + std::to_string(bvals[i]) +
                ", beta=" + std::to_string(shell_beta[i]) + ", te=" + std::to_string(shell_te[i]) +
                ", volumes=" + std::to_string(shell_volumes[i].size()));

    auto params = LoreSD::make_params_multidim(
        lmax,
        grid_size,
        reg,
        grad,
        eval_dirs,
        bvals,
        shell_volumes,
        shell_beta,
        shell_te);
    // ALS removed; no als_iters parameter
    params.init_obj_fun = have_init_obj_fun;
    params.final_obj_fun = have_final_obj_fun;

    std::unique_ptr<LoreSD::Params> recon_params;
    Eigen::MatrixXd recon_grad;
    if (have_recon_grad)
    {
        recon_grad = load_recon_grad_from_fsl(recon_bvec_path, recon_bval_path);

        std::vector<double> recon_bvals;
        std::vector<double> recon_shell_beta;
        std::vector<double> recon_shell_te;
        std::vector<std::vector<size_t>> recon_shell_volumes;

        const auto recon_b_per_vol = bvals_per_volume(recon_grad);
        const auto recon_b_groups = group_by_b(recon_b_per_vol, 50.0);
        const auto recon_beta_per_vol = expand_protocol_values(recon_beta_values, recon_b_groups, recon_b_per_vol.size(), 1.0, "recon_beta");
        const auto recon_te_per_vol = expand_protocol_values(recon_te_values, recon_b_groups, recon_b_per_vol.size(), 0.0, "recon_te");

        const auto recon_shells =
            group_shells_b_beta_te(recon_b_per_vol, recon_beta_per_vol, recon_te_per_vol, 50.0, 1e-2, 1e-2);

        recon_bvals.reserve(recon_shells.size());
        recon_shell_beta.reserve(recon_shells.size());
        recon_shell_te.reserve(recon_shells.size());
        recon_shell_volumes.reserve(recon_shells.size());
        for (const auto &sh : recon_shells)
        {
            recon_bvals.push_back(sh.b);
            recon_shell_beta.push_back(sh.beta);
            recon_shell_te.push_back(sh.te);
            recon_shell_volumes.push_back(sh.volumes);
        }

        recon_params.reset(new LoreSD::Params(
            LoreSD::make_params_multidim(
                lmax,
                grid_size,
                reg,
                recon_grad,
                eval_dirs,
                recon_bvals,
                recon_shell_volumes,
                recon_shell_beta,
                recon_shell_te)));
    }

    Header header_out(header_in);
    header_out.datatype() = DataType::Float32;
    header_out.datatype().set_byte_order_native();
    Stride::set_from_command_line(header_out, Stride::contiguous_along_axis(3, header_in));

    DWI::stash_DW_scheme(header_out, grad);
    Metadata::PhaseEncoding::clear_scheme(header_out.keyval());

    header_out.ndim() = 4;
    header_out.size(3) = MR::Math::SH::NforL(lmax);
    auto odf = Image<float>::create(argument[1], header_out);

    Header fracs_header(header_out);
    fracs_header.ndim() = 6;
    fracs_header.size(3) = grid_size[0];  // ad dimension
    fracs_header.size(4) = grid_size[1];  // rd dimension
    fracs_header.size(5) = grid_size[2];  // t2 dimension
    auto fracs = Image<float>::create(argument[2], fracs_header);

    Header response_header(header_out);
    response_header.ndim() = 5;
    response_header.size(3) = bvals.size();
    response_header.size(4) = lmax / 2 + 1;
    auto response = Image<float>::create(argument[3], response_header);

    Image<float> init_obj_fun;
    if (have_init_obj_fun)
    {
        Header init_header(header_in);
        init_header.datatype() = DataType::Float32;
        init_header.datatype().set_byte_order_native();
        init_header.ndim() = 3;
        auto path = init_obj_fun_path;
        init_obj_fun = Image<float>::create(path, init_header);
    }

    Image<float> final_obj_fun;
    if (have_final_obj_fun)
    {
        Header final_header(header_in);
        final_header.datatype() = DataType::Float32;
        final_header.datatype().set_byte_order_native();
        final_header.ndim() = 3;
        auto path = final_obj_fun_path;
        final_obj_fun = Image<float>::create(path, final_header);
    }

    Image<float> predicted_signal;
    if (have_predicted_signal)
    {
        Header predicted_header(header_in);
        predicted_header.datatype() = DataType::Float32;
        predicted_header.datatype().set_byte_order_native();
        auto path = predicted_signal_path;
        predicted_signal = Image<float>::create(path, predicted_header);
    }

    Image<float> dwi_recon;
    if (have_recon_grad)
    {
        Header recon_header(header_in);
        recon_header.datatype() = DataType::Float32;
        recon_header.datatype().set_byte_order_native();
        recon_header.ndim() = 4;
        recon_header.size(3) = recon_grad.rows();
        DWI::stash_DW_scheme(recon_header, recon_grad);
        Metadata::PhaseEncoding::clear_scheme(recon_header.keyval());
        dwi_recon = Image<float>::create(dwi_recon_path, recon_header);
    }

    auto dwi = header_in.get_image<float>().with_direct_io(3);
    LoRESD_Processor processor(params,
                               recon_params.get(),
                               mask,
                               odf,
                               fracs,
                               response,
                               dwi_recon,
                               predicted_signal,
                               init_obj_fun,
                               final_obj_fun);
    ThreadedLoop("performing LoRE-SD", dwi, 0, 3).run(processor, dwi);
    
}
