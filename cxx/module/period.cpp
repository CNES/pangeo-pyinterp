#include "pyinterp/period.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_period(py::module &m) {
  PYBIND11_NUMPY_DTYPE(pyinterp::Period, begin, last);

  py::class_<pyinterp::Period>(m, "Period")
      .def(py::init<const int64_t, const int64_t, const bool>(),
           py::arg("begin"), py::arg("last"), py::arg("within") = true)
      .def_property_readonly(
          "begin",
          [](const pyinterp::Period &self) -> int64_t { return self.begin; })
      .def_property_readonly(
          "last",
          [](const pyinterp::Period &self) -> int64_t { return self.last; })
      .def("end", &pyinterp::Period::end)
      .def("__len__",
           [](const pyinterp::Period &self) -> int64_t {
             auto result = self.length();
             if (result < 0) {
               throw std::invalid_argument("invalid period");
             }
             return result;
           })
      .def("__str__",
           [](const pyinterp::Period &self) -> std::string {
             std::stringstream ss;
             ss << "[" << self.begin << ", " << self.end() << ")";
             return ss.str();
           })
      .def("duration", &pyinterp::Period::duration)
      .def("is_null", &pyinterp::Period::is_null)
      .def(
          "__eq__",
          [](const pyinterp::Period &self, const pyinterp::Period &other) {
            return self == other;
          },
          py::arg("other"))
      .def(
          "__ne__",
          [](const pyinterp::Period &self, const pyinterp::Period &other) {
            return !(self == other);
          },
          py::arg("other"))
      .def(
          "__lt__",
          [](const pyinterp::Period &self, const pyinterp::Period &other) {
            return self.begin < other.begin;
          },
          py::arg("other"))
      .def(
          "contains",
          [](const pyinterp::Period &self, const int64_t &point) {
            return self.contains(point);
          },
          py::arg("point"))
      .def(
          "contains",
          [](const pyinterp::Period &self, const pyinterp::Period &other) {
            return self.contains(other);
          },
          py::arg("other"))
      .def("is_adjacent", &pyinterp::Period::is_adjacent, py::arg("other"))
      .def("is_before", &pyinterp::Period::is_before, py::arg("point"))
      .def("is_after", &pyinterp::Period::is_after, py::arg("point"))
      .def("intersects", &pyinterp::Period::intersects, py::arg("other"))
      .def("intersection", &pyinterp::Period::intersection, py::arg("other"))
      .def("merge", &pyinterp::Period::merge, py::arg("merge"));

  py::class_<pyinterp::PeriodList>(m, "PeriodList")
      .def(py::init<Eigen::Matrix<pyinterp::Period, -1, 1>>(),
           py::arg("periods"))
      .def(py::pickle(
          [](const pyinterp::PeriodList &self) -> py::tuple {
            return py::make_tuple(self.periods());
          },
          [](const py::tuple &state) -> pyinterp::PeriodList {
            auto periods =
                state[0].cast<Eigen::Matrix<pyinterp::Period, -1, 1>>();
            return {std::move(periods)};
          }))
      .def("__len__", &pyinterp::PeriodList::size)
      .def_property_readonly("periods", &pyinterp::PeriodList::periods)
      .def("are_periods_sorted_and_disjointed",
           &pyinterp::PeriodList::are_periods_sorted_and_disjointed,
           py::call_guard<py::gil_scoped_release>())
      .def("cross_a_period", &pyinterp::PeriodList::cross_a_period,
           py::arg("dates"), py::call_guard<py::gil_scoped_release>())
      .def("belong_to_a_period", &pyinterp::PeriodList::belong_to_a_period,
           py::arg("dates"))
      .def("is_it_close", &pyinterp::PeriodList::is_close, py::arg("period"),
           py::arg("epsilon"), py::call_guard<py::gil_scoped_release>())
      .def("join_adjacent_periods",
           &pyinterp::PeriodList::join_adjacent_periods, py::arg("epsilon"),
           py::call_guard<py::gil_scoped_release>())
      .def("within", &pyinterp::PeriodList::within, py::arg("period"),
           py::call_guard<py::gil_scoped_release>())
      .def("intersection", &pyinterp::PeriodList::intersection,
           py::arg("period"), py::call_guard<py::gil_scoped_release>())
      .def("filter", &pyinterp::PeriodList::filter, py::arg("min_duration"),
           py::call_guard<py::gil_scoped_release>())
      .def("sort", &pyinterp::PeriodList::sort,
           py::call_guard<py::gil_scoped_release>())
      .def("merge", &pyinterp::PeriodList::merge, py::arg("other"),
           py::call_guard<py::gil_scoped_release>());
}
