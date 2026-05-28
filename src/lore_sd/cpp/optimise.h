#ifndef LORE_SD_OPTIMISE_H
#define LORE_SD_OPTIMISE_H

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace lore_sd {

struct OptData {
    py::array_t<double> S;
    py::array_t<double> gaussians;
    py::array_t<double> Q;
    double reg;
    int lmax;
    int n_gaussians;
};

class Optimiser {
public:
    Optimiser(int lmax,
              int n_gaussians,
              py::array_t<double> S,
              py::array_t<double> gaussians,
              py::array_t<double> Q,
              double reg);

    std::vector<double> optimise(std::vector<double>& x);

private:
    OptData data;
};

} // namespace lore_sd

#endif // LORE_SD_OPTIMISE_H
