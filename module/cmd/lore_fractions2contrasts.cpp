/* Copyright (c) 2026.
 *
 * MRtrix3 external module command: map gaussian fractions -> contrasts
 */

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>

#include "command.h"
#include "header.h"
#include "image.h"
#include "algo/threaded_loop.h"
#include "stride.h"

using namespace MR;
using namespace App;

void usage()
{
    AUTHOR = "Siebe Leysen";
    SYNOPSIS = "Map LoRE-SD gaussian fractions to intra-/extra-axonal and free-water contrasts";

    DESCRIPTION
    +"Compute intra-axonal, extra-axonal and free-water contrasts from a LoRE-SD fractions image.";

        ARGUMENTS
        + Argument("fractions", "input fractions image (5D: x,y,z,ad,rd)").type_image_in();

        OPTIONS
        + Option("intra_axonal", "write intra-axonal contrast to file") + Argument("file").type_file_out()
        + Option("extra_axonal", "write extra-axonal contrast to file") + Argument("file").type_file_out()
        + Option("free_water", "write free-water contrast to file") + Argument("file").type_file_out()
        + Option("rfa", "write RFA map to file") + Argument("file").type_file_out()
        + Option("ad", "write AD map to file") + Argument("file").type_file_out()
        + Option("rd", "write RD map to file") + Argument("file").type_file_out()
        + Option("tc_rfa", "write tissue-conditioned RFA map excluding free-water atoms") + Argument("file").type_file_out()
        + Option("rate", "decay rate for intra-axonal weighting (default: 10)")
            + Argument("value").type_integer(10)
        + Option("with_isotropic", "treat isotropic (ad==rd) as valid for contrasts")
        + Stride::Options;
}

// Utility: linear space [0, end] with n points
static std::vector<double> linspace(double end, size_t n)
{
    std::vector<double> out(n);
    if (n == 1)
    {
        out[0] = end;
        return out;
    }
    for (size_t i = 0; i < n; ++i)
        out[i] = (static_cast<double>(i) / static_cast<double>(n - 1)) * end;
    return out;
}

static std::vector<double> normalise_0_1(const std::vector<double> &vals)
{
    auto mm = std::minmax_element(vals.begin(), vals.end());
    auto min_it = mm.first;
    auto max_it = mm.second;
    const double mn = (min_it == vals.end()) ? 0.0 : *min_it;
    const double mx = (max_it == vals.end()) ? 1.0 : *max_it;
    const double denom = (mx - mn) == 0.0 ? 1.0 : (mx - mn);
    std::vector<double> out(vals.size());
    for (size_t i = 0; i < vals.size(); ++i)
        out[i] = (vals[i] - mn) / denom;
    return out;
}

static std::vector<std::vector<double>> free_water_contrast(const std::vector<double> &ad_range, const std::vector<double> &rd_range)
{
    const size_t na = ad_range.size();
    const size_t nr = rd_range.size();
    std::vector<std::vector<double>> out(na, std::vector<double>(nr, 0.0));
    for (size_t i = 0; i < na; ++i)
    {
        for (size_t j = 0; j < nr; ++j)
        {
            const double ad = ad_range[i];
            const double rd = rd_range[j];
            if ((1000.0 * ad) >= 2.6 && (1000.0 * rd) >= 2.6 && ad >= rd)
                out[i][j] = 1.0;
        }
    }
    return out;
}

static std::vector<double> exponential_decay_function(const std::vector<double> &vals, bool reverse = false, double rate = 10.0)
{
    if (vals.empty())
        return {};
    std::vector<double> in = vals;
    if (reverse)
        std::reverse(in.begin(), in.end());

    auto n = normalise_0_1(in);
    std::vector<double> out(n.size());
    for (size_t i = 0; i < n.size(); ++i)
        out[i] = std::exp(-rate * n[i]);
    return normalise_0_1(out);
}

static std::vector<std::vector<double>> to_decay_matrix(const std::vector<double> &ad, const std::vector<double> &rd, bool axial, bool with_isotropic, double rate)
{
    const size_t na = ad.size();
    const size_t nr = rd.size();
    std::vector<std::vector<double>> unit(na, std::vector<double>(nr, 0.0));
    for (size_t i = 0; i < na; ++i)
        for (size_t j = 0; j < nr; ++j)
            unit[i][j] = (with_isotropic) ? (ad[i] >= rd[j]) : (ad[i] > rd[j]);

    std::vector<std::vector<double>> dec(na, std::vector<double>(nr, 0.0));
    if (!axial) // radial decay: function of rd, repeat across ad
    {
        auto d = exponential_decay_function(rd, false, rate);
        for (size_t i = 0; i < na; ++i)
            for (size_t j = 0; j < nr; ++j)
                dec[i][j] = d[j];
    }
    else // axial decay: function of ad, repeat across rd
    {
        auto d = exponential_decay_function(ad, false, rate);
        for (size_t i = 0; i < na; ++i)
            for (size_t j = 0; j < nr; ++j)
                dec[i][j] = d[i];
    }

    // multiply unit mask
    for (size_t i = 0; i < na; ++i)
        for (size_t j = 0; j < nr; ++j)
            dec[i][j] *= unit[i][j];

    return dec;
}

static std::vector<std::vector<double>> intra_axonal_contrast(const std::vector<double> &ad_range, const std::vector<double> &rd_range, bool with_isotropic = true, double rate = 10.0)
{
    return to_decay_matrix(ad_range, rd_range, false, with_isotropic, rate);
}

static std::vector<std::vector<double>> extra_axonal_contrast(const std::vector<double> &ad_range, const std::vector<double> &rd_range, bool with_isotropic = true, double rate = 10.0)
{
    const size_t na = ad_range.size();
    const size_t nr = rd_range.size();
    auto fw = free_water_contrast(ad_range, rd_range);
    auto ia = intra_axonal_contrast(ad_range, rd_range, with_isotropic, rate);
    std::vector<std::vector<double>> out(na, std::vector<double>(nr, 0.0));
    for (size_t i = 0; i < na; ++i)
    {
        for (size_t j = 0; j < nr; ++j)
        {
            out[i][j] = 1.0 - fw[i][j] - ia[i][j];
            if (ad_range[i] < rd_range[j])
                out[i][j] = 0.0;
        }
    }
    return out;
}

static std::vector<std::vector<double>> rfa_map(const std::vector<double> &ad, const std::vector<double> &rd)
{
    const size_t na = ad.size();
    const size_t nr = rd.size();
    std::vector<std::vector<double>> out(na, std::vector<double>(nr, 0.0));
    for (size_t i = 0; i < na; ++i)
    {
        for (size_t j = 0; j < nr; ++j)
        {
            double adv = ad[i];
            double rdv = rd[j];
            if (!(adv >= rdv))
            {
                out[i][j] = 0.0;
                continue;
            }
            double lambda_mean = (adv + 2.0 * rdv) / 3.0;
            double num = std::sqrt((adv - lambda_mean) * (adv - lambda_mean) + 2.0 * (rdv - lambda_mean) * (rdv - lambda_mean));
            double den = std::sqrt(adv * adv + 2.0 * rdv * rdv);
            if (den == 0.0)
                out[i][j] = 0.0;
            else
                out[i][j] = std::sqrt(3.0 / 2.0) * (num / den);
        }
    }
    return out;
}

static std::vector<std::vector<double>> ad_map(const std::vector<double> &ad, const std::vector<double> &rd)
{
    const size_t na = ad.size();
    const size_t nr = rd.size();
    std::vector<std::vector<double>> out(na, std::vector<double>(nr, 0.0));
    for (size_t i = 0; i < na; ++i)
    {
        for (size_t j = 0; j < nr; ++j)
        {
            double adv = ad[i];
            double rdv = rd[j];
            if (!(adv >= rdv))
            {
                out[i][j] = 0.0;
                continue;
            }
            out[i][j] = adv;
        }
    }
    return out;
}

static std::vector<std::vector<double>> rd_map(const std::vector<double> &ad, const std::vector<double> &rd)
{
    const size_t na = ad.size();
    const size_t nr = rd.size();
    std::vector<std::vector<double>> out(na, std::vector<double>(nr, 0.0));
    for (size_t i = 0; i < na; ++i)
    {
        for (size_t j = 0; j < nr; ++j)
        {
            double adv = ad[i];
            double rdv = rd[j];
            if (!(adv >= rdv))
            {
                out[i][j] = 0.0;
                continue;
            }
            out[i][j] = rdv;
        }
    }
    return out;
}

static double get_contrast(const std::vector<float> &fs, const std::vector<double> &ad, const std::vector<double> &rd, const std::vector<std::vector<double>> &weights)
{
    // fs is flattened with ad outer, rd inner: idx = i * nr + j
    const size_t na = ad.size();
    const size_t nr = rd.size();
    double sum = 0.0;
    for (size_t i = 0; i < na; ++i)
    {
        for (size_t j = 0; j < nr; ++j)
        {
            const size_t idx = i * nr + j;
            const double w = weights[i][j];
            const double f = (idx < fs.size()) ? static_cast<double>(fs[idx]) : 0.0;
            sum += w * f;
        }
    }
    return sum;
}

static double get_conditional_contrast(
    const std::vector<float> &fs,
    const std::vector<double> &ad,
    const std::vector<double> &rd,
    const std::vector<std::vector<double>> &values,
    const std::vector<std::vector<double>> &condition)
{
    // Computes sum f * condition * value / sum f * condition
    // fs is flattened with ad outer, rd inner: idx = i * nr + j

    const size_t na = ad.size();
    const size_t nr = rd.size();

    double num = 0.0;
    double den = 0.0;

    for (size_t i = 0; i < na; ++i)
    {
        for (size_t j = 0; j < nr; ++j)
        {
            const size_t idx = i * nr + j;

            const double f = (idx < fs.size()) ? static_cast<double>(fs[idx]) : 0.0;
            const double c = condition[i][j];
            const double v = values[i][j];

            num += f * c * v;
            den += f * c;
        }
    }

    return (den > 0.0) ? (num / den) : 0.0;
}

static std::vector<std::vector<double>> non_free_water_contrast(
    const std::vector<double> &ad_range,
    const std::vector<double> &rd_range)
{
    auto fw = free_water_contrast(ad_range, rd_range);

    const size_t na = ad_range.size();
    const size_t nr = rd_range.size();

    std::vector<std::vector<double>> out(na, std::vector<double>(nr, 0.0));

    for (size_t i = 0; i < na; ++i)
        for (size_t j = 0; j < nr; ++j)
            out[i][j] = 1.0 - fw[i][j];

    return out;
}

class ContrastsProcessor
{
public:
    ContrastsProcessor(Image<float> &fractions, Image<float> &out_intra, Image<float> &out_extra, Image<float> &out_free, Image<float> &out_rfa, 
        Image<float> &out_ad, Image<float> &out_rd, Image<float> &out_tc_rfa, int rate, bool with_isotropic)
        : fractions_image(fractions), intra_image(out_intra), extra_image(out_extra), 
            free_image(out_free), rfa_image(out_rfa), ad_image(out_ad), rd_image(out_rd), tc_rfa_image(out_tc_rfa), rate(rate), with_isotropic(with_isotropic)
    {
        // nothing
    }

    void operator()(Image<float> &fracs)
    {
        // read dimensions
        const size_t na = static_cast<size_t>(fracs.size(3));
        const size_t nr = static_cast<size_t>(fracs.size(4));

        std::vector<float> fs;
        fs.reserve(na * nr);
        for (size_t i = 0; i < na; ++i)
        {
            fracs.index(3) = static_cast<int>(i);
            for (size_t j = 0; j < nr; ++j)
            {
                fracs.index(4) = static_cast<int>(j);
                fs.push_back(fracs.value());
            }
        }

        // build ranges
        auto ad_range = linspace(3.3e-3, na);
        auto rd_range = linspace(3.3e-3, nr);

        auto intra = intra_axonal_contrast(ad_range, rd_range, with_isotropic, static_cast<double>(rate));
        auto extra = extra_axonal_contrast(ad_range, rd_range, with_isotropic, static_cast<double>(rate));
        auto freew = free_water_contrast(ad_range, rd_range);
        auto rfam = rfa_map(ad_range, rd_range);
        auto ad = ad_map(ad_range, rd_range);
        auto rd = rd_map(ad_range, rd_range);
        auto tc_rfa = non_free_water_contrast(ad_range, rd_range);
        const double intra_val = get_contrast(fs, ad_range, rd_range, intra);
        const double extra_val = get_contrast(fs, ad_range, rd_range, extra);
        const double free_val = get_contrast(fs, ad_range, rd_range, freew);
        const double rfa_val = get_contrast(fs, ad_range, rd_range, rfam);
        const double ad_val = get_contrast(fs, ad_range, rd_range, ad);
        const double rd_val = get_contrast(fs, ad_range, rd_range, rd);
        const double tc_rfa_val = get_conditional_contrast(fs, ad_range, rd_range, rfam, tc_rfa);

        if (intra_image.valid())
        {
            assign_pos_of(fracs, 0, 3).to(intra_image);
            intra_image.value() = static_cast<float>(intra_val);
        }
        if (extra_image.valid())
        {
            assign_pos_of(fracs, 0, 3).to(extra_image);
            extra_image.value() = static_cast<float>(extra_val);
        }
        if (free_image.valid())
        {
            assign_pos_of(fracs, 0, 3).to(free_image);
            free_image.value() = static_cast<float>(free_val);
        }
        if (rfa_image.valid())
        {
            assign_pos_of(fracs, 0, 3).to(rfa_image);
            rfa_image.value() = static_cast<float>(rfa_val);
        }
        if (ad_image.valid())
        {
            assign_pos_of(fracs, 0, 3).to(ad_image);
            ad_image.value() = static_cast<float>(ad_val);
        }
        if (rd_image.valid())
        {
            assign_pos_of(fracs, 0, 3).to(rd_image);
            rd_image.value() = static_cast<float>(rd_val);
        }
        if (tc_rfa_image.valid())
        {
            assign_pos_of(fracs, 0, 3).to(tc_rfa_image);
            tc_rfa_image.value() = static_cast<float>(tc_rfa_val);
        }
    }

private:
    Image<float> fractions_image;
    Image<float> intra_image;
    Image<float> extra_image;
    Image<float> free_image;
    Image<float> rfa_image;
    Image<float> ad_image;
    Image<float> rd_image;
    Image<float> tc_rfa_image;
    int rate = 10;
    bool with_isotropic = true;
};

void run()
{
    auto header_in = Header::open(argument[0]);
    auto fractions = header_in.get_image<float>().with_direct_io(3);

    if (fractions.ndim() != 5)
        throw Exception("Input fractions image must be 5D (x,y,z,ad,rd)");

    auto opt = get_options("rate");
    int rate = 10;
    if (opt.size())
        rate = to<int>(opt[0][0]);

    opt = get_options("with_isotropic");
    bool with_iso = opt.size() ? true : false;

    // parse optional output paths
    auto opt_intra = get_options("intra_axonal");
    auto opt_extra = get_options("extra_axonal");
    auto opt_free = get_options("free_water");
    auto opt_rfa = get_options("rfa");
    auto opt_ad = get_options("ad");
    auto opt_rd = get_options("rd");
    auto opt_tc_rfa = get_options("tc_rfa");

    const bool write_intra = opt_intra.size() != 0;
    const bool write_extra = opt_extra.size() != 0;
    const bool write_free = opt_free.size() != 0;
    const bool write_rfa = opt_rfa.size() != 0;
    const bool write_ad = opt_ad.size() != 0;
    const bool write_rd = opt_rd.size() != 0;
    const bool write_tc_rfa = opt_tc_rfa.size() != 0;

    auto mkdir_parent = [](const std::string &path){
        const auto pos = path.find_last_of('/');
        if (pos == std::string::npos) return;
        const std::string dir = path.substr(0, pos);
        std::string cmd = std::string("mkdir -p '") + dir + "'";
        int r = std::system(cmd.c_str()); (void)r;
    };

    // prepare headers for outputs: 3D float images
    Header out_header(header_in);
    out_header.datatype() = DataType::Float32;
    out_header.datatype().set_byte_order_native();
    out_header.ndim() = 3;

    Image<float> intra_img;
    Image<float> extra_img;
    Image<float> free_img;
    Image<float> rfa_img;
    Image<float> ad_img;
    Image<float> rd_img;
    Image<float> tc_rfa_img;

    if (write_intra)
    {
        const std::string intra_path = std::string(opt_intra[0][0]);
        mkdir_parent(intra_path);
        intra_img = Image<float>::create(intra_path, out_header);
    }
    if (write_extra)
    {
        const std::string extra_path = std::string(opt_extra[0][0]);
        mkdir_parent(extra_path);
        extra_img = Image<float>::create(extra_path, out_header);
    }
    if (write_free)
    {
        const std::string free_path = std::string(opt_free[0][0]);
        mkdir_parent(free_path);
        free_img = Image<float>::create(free_path, out_header);
    }
    if (write_rfa)
    {
        const std::string rfa_path = std::string(opt_rfa[0][0]);
        mkdir_parent(rfa_path);
        rfa_img = Image<float>::create(rfa_path, out_header);
    }
    if (write_ad)
    {
        const std::string ad_path = std::string(opt_ad[0][0]);
        mkdir_parent(ad_path);
        ad_img = Image<float>::create(ad_path, out_header);
    }
    if (write_rd)
    {
        const std::string rd_path = std::string(opt_rd[0][0]);
        mkdir_parent(rd_path);
        rd_img = Image<float>::create(rd_path, out_header);
    }

    if (write_tc_rfa)
    {
        const std::string tc_rfa_path = std::string(opt_tc_rfa[0][0]);
        mkdir_parent(tc_rfa_path);
        tc_rfa_img = Image<float>::create(tc_rfa_path, out_header);
    }


    ContrastsProcessor proc(fractions, intra_img, extra_img, free_img, rfa_img, ad_img, rd_img, tc_rfa_img, rate, with_iso);
    ThreadedLoop("mapping fractions -> contrasts", fractions, 0, 3).run(proc, fractions);
}
