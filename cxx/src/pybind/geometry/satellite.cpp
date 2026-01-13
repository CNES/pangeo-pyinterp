// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/geometry/satellite.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/satellite/algorithms/crossover.hpp"
#include "pyinterp/geometry/satellite/transforms/swath.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry {
namespace satellite::pybind {

constexpr auto kFindCrossoverDoc = R"doc(
Find crossover point between two satellite half-orbits.

Args:
    lon1: Longitude array of the first half-orbit in degrees.
    lat1: Latitude array of the first half-orbit in degrees.
    lon2: Longitude array of the second half-orbit in degrees.
    lat2: Latitude array of the second half-orbit in degrees.
    predicate: Maximum acceptable distance to consider a vertex as nearest.
    allow_multiple: If true, all crossover points are returned;
        otherwise, only a unique crossover point is searched.
    use_cartesian: If true, the intersection search is performed on a Cartesian
        plane; otherwise, geodetic calculations are used.
    strategy: Calculation strategy.
    spheroid: Optional spheroid for geodetic calculations.

Returns:
    All crossover points found that pass the predicate filter.

Raises:
    RuntimeError: If ``allow_multiple`` is false and multiple crossover points
        are found.

Note:
    If ``use_cartesian`` is true, the intersection search is performed on a
    Cartesian plane, which provides faster results than geodetic calculations.
    However, this approach is only appropriate when an approximate determination
    is acceptable. The accuracy depends on the vertices of the linestrings
    being close to each other; if they are widely spaced, the determined
    geographical point may be significantly incorrect due to Cartesian
    approximation errors.
)doc";

constexpr auto kCalculateSwathDoc = R"doc(
Calculate the swath coordinates from the nadir coordinates.

Args:
    lon_nadir: Longitude of nadir in degrees.
    lat_nadir: Latitude of nadir in degrees.
    delta_ac: Across-track distance between two consecutive pixels (meters).
    half_gap: Half of the gap between the nadir and the first pixel (meters).
    half_swath: Half of the swath width (pixels).
    spheroid: Optional spheroid model.

Returns:
    Tuple of longitude and latitude matrices of the swath.
)doc";

inline auto init_crossover(nb::module_& m) -> void {
  nb::class_<algorithms::CrossoverResult>(m, "CrossoverResult",
                                          "Result of a crossover detection")
      .def_prop_ro(
          "point",
          [](const algorithms::CrossoverResult& self) -> geographic::Point {
            return self.point;
          },
          "The crossover point")
      .def_prop_ro(
          "index1",
          [](const algorithms::CrossoverResult& self) -> size_t {
            return self.index1;
          },
          "Index of nearest vertex in first linestring")
      .def_prop_ro(
          "index2",
          [](const algorithms::CrossoverResult& self) -> size_t {
            return self.index2;
          },
          "Index of nearest vertex in second linestring");

  m.def("find_crossovers", &algorithms::find_crossovers, "lon1"_a, "lat1"_a,
        "lon2"_a, "lat2"_a, "predicate"_a, "allow_multiple"_a = false,
        "use_cartesian"_a = true,
        "strategy"_a = geographic::StrategyMethod::kVincenty,
        "spheroid"_a = std::nullopt, kFindCrossoverDoc,
        nb::call_guard<nb::gil_scoped_release>());
}

inline auto init_algorithms(nb::module_& m) -> void { init_crossover(m); }

inline auto init_transforms(nanobind::module_& m) -> void {
  m.def("calculate_swath", &swath::calculate<double>, "lon_nadir"_a,
        "lat_nadir"_a, "delta_ac"_a, "half_gap"_a, "half_swath"_a,
        "spheroid"_a = std::nullopt, kCalculateSwathDoc,
        nb::call_guard<nb::gil_scoped_release>());
}

}  // namespace satellite::pybind

namespace pybind {

auto init_satellite(nanobind::module_& m) -> void {
  auto submodule = m.def_submodule("satellite", "Satellite geometry");
  satellite::pybind::init_crossover(submodule);
  satellite::pybind::init_transforms(submodule);
}

}  // namespace pybind

}  // namespace pyinterp::geometry
