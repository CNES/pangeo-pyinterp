// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/geometry/satellite.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <format>
#include <string>

#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/satellite/algorithms/crossover.hpp"
#include "pyinterp/geometry/satellite/algorithms/track_decomposition.hpp"
#include "pyinterp/geometry/satellite/transforms/swath.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry {
namespace satellite::pybind {

// Convert a string strategy name to the enum.
constexpr auto parse_strategy(std::string const& name)
    -> algorithms::DecompositionStrategy {
  if (name == "latitude_bands" || name == "bands") {
    return algorithms::DecompositionStrategy::kLatitudeBands;
  }
  if (name == "monotonic" || name == "monotonic_segments") {
    return algorithms::DecompositionStrategy::kMonotonicSegments;
  }
  throw std::invalid_argument("Unknown strategy: '" + name +
                              "'. Use 'latitude_bands' or 'monotonic'.");
}

// Convert a LatitudeZone enum to a human-readable string.
constexpr auto zone_to_string(algorithms::LatitudeZone z) -> char const* {
  switch (z) {
    case algorithms::LatitudeZone::kSouth:
      return "south";
    case algorithms::LatitudeZone::kMid:
      return "mid";
    case algorithms::LatitudeZone::kNorth:
      return "north";
  }
  return "unknown";
}

// Convert an OrbitDirection enum to a human-readable string.
constexpr auto orbit_to_string(algorithms::OrbitDirection d) -> char const* {
  switch (d) {
    case algorithms::OrbitDirection::kPrograde:
      return "prograde";
    case algorithms::OrbitDirection::kRetrograde:
      return "retrograde";
  }
  return "unknown";
}

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

inline auto init_track_decomposition(nanobind::module_& m) -> void {
  nb::enum_<algorithms::LatitudeZone>(
      m, "LatitudeZone", "Latitude-band classification of a track segment.")
      .value("SOUTH", algorithms::LatitudeZone::kSouth, "lat <= south_limit")
      .value("MID", algorithms::LatitudeZone::kMid,
             "south_limit < lat < north_limit")
      .value("NORTH", algorithms::LatitudeZone::kNorth, "lat >= north_limit");

  nb::enum_<algorithms::OrbitDirection>(
      m, "OrbitDirection", "Orbital direction inferred from longitude.")
      .value("PROGRADE", algorithms::OrbitDirection::kPrograde,
             "Predominantly eastward.")
      .value("RETROGRADE", algorithms::OrbitDirection::kRetrograde,
             "Predominantly westward (also tie case).");

  nb::class_<algorithms::DecompositionOptions>(
      m, "DecompositionOptions", "Options for track decomposition")
      .def(nb::init<>(),
           "Create a DecompositionOptions instance with default values.")
      .def_prop_ro(
          "swath_width_km",
          [](const algorithms::DecompositionOptions& self) {
            return self.swath_width_km();
          },
          "Cross-track swath full width in kilometers.  ``0`` means "
          "line-string mode (nadir track only).")
      .def_prop_ro(
          "south_limit",
          [](const algorithms::DecompositionOptions& self) {
            return self.south_limit();
          },
          "Southern latitude boundary for band classification.")
      .def_prop_ro(
          "north_limit",
          [](const algorithms::DecompositionOptions& self) {
            return self.north_limit();
          },
          "Northern latitude boundary for band classification.")
      .def_prop_ro(
          "min_edge_size",
          [](const algorithms::DecompositionOptions& self) {
            return self.min_edge_size();
          },
          "Minimum number of points for the first/last segment before "
          "edge-merge is applied.")
      .def_prop_ro(
          "merge_area_ratio",
          [](const algorithms::DecompositionOptions& self) {
            return self.merge_area_ratio();
          },
          "Area-based merge threshold for the monotonic strategy.  Set to "
          "``0`` to disable area-based merging entirely.")
      .def_prop_ro(
          "max_segments",
          [](const algorithms::DecompositionOptions& self) {
            return self.max_segments();
          },
          "Maximum number of output segments.  ``0`` means unlimited.")
      .def(
          "with_swath_width_km",
          [](algorithms::DecompositionOptions self, double swath_width_km) {
            return self.with_swath_width_km(swath_width_km);
          },
          "swath_width_km"_a,
          "Update the swath width option and return a new instance.")
      .def(
          "with_south_limit",
          [](algorithms::DecompositionOptions self, double south_limit) {
            return self.with_south_limit(south_limit);
          },
          "south_limit"_a,
          "Update the south limit option and return a new instance.")
      .def(
          "with_north_limit",
          [](algorithms::DecompositionOptions self, double north_limit) {
            return self.with_north_limit(north_limit);
          },
          "north_limit"_a,
          "Update the north limit option and return a new instance.")
      .def(
          "with_min_edge_size",
          [](algorithms::DecompositionOptions self, size_t min_edge_size) {
            return self.with_min_edge_size(min_edge_size);
          },
          "min_edge_size"_a,
          "Update the minimum edge size option and return a new instance.")
      .def(
          "with_merge_area_ratio",
          [](algorithms::DecompositionOptions self, double merge_area_ratio) {
            return self.with_merge_area_ratio(merge_area_ratio);
          },
          "merge_area_ratio"_a,
          "Update the merge area ratio option and return a new instance.")
      .def(
          "with_max_segments",
          [](algorithms::DecompositionOptions self, size_t max_segments) {
            return self.with_max_segments(max_segments);
          },
          "max_segments"_a,
          "Update the maximum segments option and return a new instance.");

  nb::class_<algorithms::TrackSegment>(
      m, "TrackSegment",
      "A contiguous segment of the track with its "
      "bounding box and metadata.")
      .def_ro("first_index", &algorithms::TrackSegment::first_index,
              "First index in the input arrays (inclusive).")
      .def_ro("last_index", &algorithms::TrackSegment::last_index,
              "Last index in the input arrays (inclusive).")
      .def_ro("bbox", &algorithms::TrackSegment::bbox,
              "Bounding box (swath-expanded if applicable).")
      .def_ro("zone", &algorithms::TrackSegment::zone,
              "Dominant latitude zone of this segment.")
      .def_ro("orbit", &algorithms::TrackSegment::orbit,
              "Orbit direction (shared across all segments).")
      .def_prop_ro(
          "size",
          [](algorithms::TrackSegment const& self) { return self.size(); },
          "Number of points in this segment.")
      .def("__repr__", [](algorithms::TrackSegment const& self) {
        return std::format(
            "TrackSegment([{}, {}], zone={}, orbit={}, "
            "bbox=lon[{:.2f},{:.2f}] lat[{:.2f},{:.2f}])",
            self.first_index, self.last_index, zone_to_string(self.zone),
            orbit_to_string(self.orbit), self.bbox.min_corner().lon(),
            self.bbox.max_corner().lon(), self.bbox.min_corner().lat(),
            self.bbox.max_corner().lat());
      });

  m.def(
      "decompose_track",
      [](const Eigen::Ref<const Eigen::VectorXd>& lon,
         const Eigen::Ref<const Eigen::VectorXd>& lat,
         const std::optional<std::string>& strategy,
         const std::optional<algorithms::DecompositionOptions>& opts)
          -> std::vector<algorithms::TrackSegment> {
        return algorithms::decompose_track(
            lon, lat,
            strategy ? parse_strategy(*strategy)
                     : algorithms::DecompositionStrategy::kLatitudeBands,
            opts ? *opts : algorithms::DecompositionOptions{});
      },
      "lon"_a, "lat"_a, "strategy"_a = std::nullopt, "opts"_a = std::nullopt,
      R"doc(Decompose a satellite track into segments with tight bounding boxes.

Args:
    lon : Longitude array (degrees), 1-D.
    lat : Latitude array (degrees), 1-D. Must have the same size as *lon*.
    strategy : Decomposition strategy.
        - 'latitude_bands' (default): Decompose the track into latitude bands
           defined by the south and north limits in *opts*.  This strategy is
           generally faster and produces more compact segments, but may result
           in segments that are not strictly monotonic in latitude.
        - 'monotonic': Decompose the track into segments that are monotonic in
           latitude, then merge adjacent segments based on the area of the
           bounding boxes and the *merge_area_ratio* in *opts*.  This strategy
           produces segments that are strictly monotonic in latitude, which can
           be beneficial for certain applications, but may result in more
           segments and longer processing time.
    opts : DecompositionOptions, optional

Returns:
    Ordered, contiguous, non-overlapping segments whose union covers the
    entire input track.

Raises:
    ValueError: If inputs are empty, have different sizes, or limits are inverted.
)doc",
      nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "infer_orbit_direction", &algorithms::infer_orbit_direction, "lon"_a,
      R"doc(Infer the orbital direction (prograde or retrograde) from the longitude array of a track.
Args:
    lon : Longitude array (degrees), 1-D.
Returns:
    OrbitDirection enum value indicating the inferred orbital direction.
)doc",
      nb::call_guard<nb::gil_scoped_release>());

  m.def(
      "filter_by_extent",
      [](std::vector<algorithms::TrackSegment> const& segments,
         geometry::geographic::Box const& extent)
          -> std::vector<algorithms::TrackSegment> {
        return filter_by_extent(segments, extent);
      },
      "segments"_a, "extent"_a,
      R"doc(Filter segments by intersection with a data extent.

Args:
    segments : List of TrackSegment objects to filter.
    extent : Geographic bounding box to use for filtering.

Returns:
    List of TrackSegment objects whose bounding boxes intersect with the given extent.
)doc",
      nb::call_guard<nb::gil_scoped_release>());
}

}  // namespace satellite::pybind

namespace pybind {

auto init_satellite(nanobind::module_& m) -> void {
  auto submodule = m.def_submodule("satellite", "Satellite geometry");
  satellite::pybind::init_crossover(submodule);
  satellite::pybind::init_track_decomposition(submodule);
  satellite::pybind::init_transforms(submodule);
}

}  // namespace pybind

}  // namespace pyinterp::geometry
