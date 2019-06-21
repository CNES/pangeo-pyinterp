#include "pyinterp/detail/gsl/error_handler.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern void init_axis(py::module&);
extern void init_cartesian(py::module&);
extern void init_geodetic(py::module&);

PYBIND11_MODULE(core, m) {
  m.doc() = R"__doc__(
Core module
-----------
)__doc__";

  pyinterp::detail::gsl::set_error_handler();

  auto cartesian = m.def_submodule("cartesian", R"__doc__(
Interpolation in cartesian space
--------------------------------
)__doc__");

  auto geodetic = m.def_submodule("geodetic", R"__doc__(
Interpolation in spherical equatorial coordinate system in degree
-----------------------------------------------------------------
)__doc__");

  init_axis(m);
  init_cartesian(cartesian);
  init_geodetic(geodetic);
}