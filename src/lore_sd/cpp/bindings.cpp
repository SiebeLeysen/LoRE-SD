#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "optimise.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_optimiser, m) {
    py::class_<lore_sd::Optimiser>(m, "Optimiser")
        .def(py::init<int, int, py::array_t<double>, py::array_t<double>, py::array_t<double>, double>())
        .def("optimise", &lore_sd::Optimiser::optimise);
}
