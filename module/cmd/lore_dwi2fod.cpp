/* Copyright (c) 2026.
 *
 * External MRtrix3 module for LoRE-SD.
 */

#include <cmath>


#include "command.h"
#include "header.h"
#include "image.h"
#include "algo/threaded_loop.h"
#include "dwi/gradient.h"
#include "dwi/shells.h"
#include "dwi/directions/predefined.h"
#include "math/SH.h"
#include "metadata/phase_encoding.h"
#include "stride.h"

#include "lore_sd/lore_sd.h"

using namespace MR;
using namespace App;

void usage()
{
    AUTHOR = "Siebe Leysen";

    SYNOPSIS = "Estimate fibre orientation distributions using LoRE-SD";

    DESCRIPTION
    +"This is a standalone LoRE-SD implementation built as an MRtrix3 external module.";

    ARGUMENTS
    +Argument("dwi", "the input diffusion-weighted image").type_image_in() + Argument("odf", "output ODF image").type_image_out() + Argument("fracs", "output LoRE-SD fractions image").type_image_out() + Argument("response", "output LoRE-SD response image").type_image_out();

    OPTIONS
    +DWI::GradImportOptions() + DWI::ShellsOption 
    + Option("mask", "only perform computation within the specified binary brain mask image")
     + Argument("image").type_image_in()
    + Option("lmax", "maximum spherical harmonic order (default: 8)")
     + Argument("order").type_integer(8)
    + Option("grid_size", "grid size for Da/Dr (default: 10)")
     + Argument("size").type_integer(10)
    + Option("reg", "regularisation parameter (default: 1e-3)")
     + Argument("value").type_float(1e-3)
    + Option("maxeval", "max evaluations for main optimizer (default: 400") 
     + Argument("evals").type_integer(400) 
    + Option("init_obj_fun", "write initial objective function per voxel to an output image")
     + Argument("image").type_image_out()
    + Option("final_obj_fun", "write final objective function per voxel to an output image")
     + Argument("image").type_image_out()
    + Option("predicted_signal", "write the DWI signal predicted by LoRE-SD to an output image")
     + Argument("image").type_image_out() + Stride::Options;
}

class LoRESD_Processor
{
    MEMALIGN(LoRESD_Processor)
public:
    LoRESD_Processor(const LoreSD::Params &params,
                     Image<bool> &mask,
                     Image<float> &odf,
                     Image<float> &fracs,
                     Image<float> &response,
                     Image<float> &predicted_signal,
                     Image<float> &init_obj_fun,
                     Image<float> &final_obj_fun) : params(params),
                                                                mask(mask),
                                                                odf_image(odf),
                                                                fracs_image(fracs),
                                                                response_image(response),
                                                                predicted_signal_image(predicted_signal),
                                                                init_obj_fun_image(init_obj_fun),
                                                                final_obj_fun_image(final_obj_fun),
                                                                dwi_data(0)
    {
        n_odf = MR::Math::SH::NforL(params.lmax);
        n_fracs = static_cast<size_t>(params.grid_size * params.grid_size);
        n_response = static_cast<size_t>(params.bvals.size()) * static_cast<size_t>(params.lmax / 2 + 1);
        n_predicted_signal = 0;
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

        auto result = LoreSD::fit_voxel(dwi_data, params);

        write_vector(dwi, odf_image, result.odf, n_odf);
        write_vector(dwi, fracs_image, result.fracs, n_fracs);
        write_vector(dwi, response_image, result.response, n_response);
        write_vector(dwi, predicted_signal_image, result.predicted_signal, n_predicted_signal);
        write_scalar(dwi, init_obj_fun_image, result.f0);
        write_scalar(dwi, final_obj_fun_image, result.f1);

        (void)result; // no CSV profiling in public release
    }

private:
    LoreSD::Params params;
    Image<bool> mask;
    Image<float> odf_image;
    Image<float> fracs_image;
    Image<float> response_image;
    Image<float> predicted_signal_image;
    Image<float> init_obj_fun_image;
    Image<float> final_obj_fun_image;
    Eigen::VectorXd dwi_data;
    size_t n_odf = 0;
    size_t n_fracs = 0;
    size_t n_response = 0;
    size_t n_predicted_signal = 0;

    void write_zero_outputs(Image<float> &dwi)
    {
        static const std::vector<float> empty;
        write_vector(dwi, odf_image, empty, n_odf);
        write_vector(dwi, fracs_image, empty, n_fracs);
        write_vector(dwi, response_image, empty, n_response);
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

    int grid_size = 10;
    opt = get_options("grid_size");
    if (opt.size())
        grid_size = to<int>(opt[0][0]);

    double reg = 1e-3;
    opt = get_options("reg");
    if (opt.size())
        reg = to<double>(opt[0][0]);

    int maxeval = 400;
    opt = get_options("maxeval");
    if (opt.size())
        maxeval = to<int>(opt[0][0]);

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

    // profiling removed in public release

    std::vector<double> bvals;
    std::vector<std::vector<size_t>> shell_volumes;
    
    DWI::Shells shells(grad);
    bvals.reserve(shells.count());
    shell_volumes.reserve(shells.count());
    for (size_t i = 0; i < shells.count(); ++i)
    {
        bvals.push_back(shells[i].get_mean());
        shell_volumes.push_back(shells[i].get_volumes());
    }
        

    auto params = LoreSD::make_params(lmax, grid_size, reg, grad, eval_dirs, bvals, shell_volumes);
    // ALS removed; no als_iters parameter
    params.maxeval = maxeval;
    params.init_obj_fun = have_init_obj_fun;
    params.final_obj_fun = have_final_obj_fun;

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
    fracs_header.ndim() = 5;
    fracs_header.size(3) = grid_size;  // ad dimension
    fracs_header.size(4) = grid_size;  // rd dimension
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

    auto dwi = header_in.get_image<float>().with_direct_io(3);
    LoRESD_Processor processor(params, mask, odf, fracs, response, predicted_signal, init_obj_fun, final_obj_fun);
    ThreadedLoop("performing LoRE-SD", dwi, 0, 3).run(processor, dwi);
    
}
