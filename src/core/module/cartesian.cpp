#include "pyinterp/bivariate.hpp"
#include "pyinterp/trivariate.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace geometry = pyinterp::detail::geometry;

void init_cartesian(py::module& m) {
  pyinterp::init_bivariate<geometry::Point2D, double>(m);
  pyinterp::init_trivariate<geometry::Point3D, double>(m);
}
