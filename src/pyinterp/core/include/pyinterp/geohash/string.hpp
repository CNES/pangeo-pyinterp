// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <algorithm>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/geodetic/box.hpp"
#include "pyinterp/geodetic/multipolygon.hpp"
#include "pyinterp/geodetic/point.hpp"
#include "pyinterp/geodetic/polygon.hpp"

namespace pyinterp::geohash::string {

/// Handle the numpy arrays of geohash.
class Array {
 public:
  /// Creation of a vector of "size" items of strings of maximum length
  /// "precision"
  Array(const size_t size, const uint32_t precision)
      : array_(new std::vector<char>(size * precision, '\0')),
        capsule_(
            array_,
            [](void *ptr) { delete static_cast<std::vector<char> *>(ptr); }),
        chars_(precision),
        size_(size) {}

  /// Resize the array to the given size.
  auto resize(const size_t size) -> void {
    array_->resize(size * chars_);
    size_ = size;
  }

  /// Get the pointer to the raw memory
  [[nodiscard]] inline auto buffer() const -> char * { return array_->data(); }

  /// Creates the numpy array from the memory allocated in the C++ code without
  /// copying the data.
  [[nodiscard]] inline auto pyarray() -> pybind11::array {
    return pybind11::array(pybind11::dtype("S" + std::to_string(chars_)),
                           {size_}, {chars_ * sizeof(char)}, array_->data(),
                           capsule_);
  }

  static auto get_info(const pybind11::array &hash, pybind11::ssize_t ndim)
      -> pybind11::buffer_info;

 private:
  std::vector<char> *array_;
  pybind11::capsule capsule_;
  uint32_t chars_;
  size_t size_;
};

/// Allocates a numpy array of strings of maximum length "precision"
auto allocate_array(size_t size, uint32_t precision) -> Array;

/// Encode a point into geohash with the given bit depth
auto encode(const geodetic::Point &point, char *buffer, uint32_t precision)
    -> void;

/// Encode points into geohash with the given bit depth
[[nodiscard]] auto encode(const Eigen::Ref<const Eigen::VectorXd> &lon,
                          const Eigen::Ref<const Eigen::VectorXd> &lat,
                          uint32_t precision) -> pybind11::array;

/// Returns the region encoded
[[nodiscard]] auto bounding_box(const char *hash, size_t count)
    -> geodetic::Box;

/// Decode a hash into a spherical equatorial point. If round is true, the
/// coordinates of the points will be rounded to the accuracy defined by the
/// GeoHash.
[[nodiscard]] auto decode(const char *hash, size_t count, bool round)
    -> geodetic::Point;

/// Decode hashes into a spherical equatorial points. If round is true, the
/// coordinates of the points will be rounded to the accuracy defined by the
/// GeoHash.
[[nodiscard]] auto decode(const pybind11::array &hash, bool round)
    -> std::tuple<Eigen::VectorXd, Eigen::VectorXd>;

/// Returns all neighbors hash clockwise from north around northwest at the
/// given precision:
///   7 0 1
///   6 x 2
///   5 4 3
[[nodiscard]] auto neighbors(const char *hash, size_t count) -> pybind11::array;

/// Returns the area covered by the GeoHash
[[nodiscard]] inline auto area(const char *const hash, size_t count,
                               const std::optional<geodetic::Spheroid> &wgs)
    -> double {
  return bounding_box(hash, count).area(wgs);
}

/// Returns the area covered by the GeoHash codes
[[nodiscard]] auto area(const pybind11::array &hash,
                        const std::optional<geodetic::Spheroid> &wgs)
    -> Eigen::MatrixXd;

/// Returns all GeoHash within the given region
[[nodiscard]] auto bounding_boxes(const std::optional<geodetic::Box> &box,
                                  uint32_t precision) -> pybind11::array;

[[nodiscard]] auto bounding_boxes(const geodetic::Polygon &polygon,
                                  uint32_t precision, size_t num_threads)
    -> pybind11::array;

[[nodiscard]] auto bounding_boxes(const geodetic::MultiPolygon &polygons,
                                  uint32_t precision, size_t num_threads)
    -> pybind11::array;

/// Returns the start and end indexes of the different GeoHash boxes.
[[nodiscard]] auto where(const pybind11::array &hash) -> std::unordered_map<
    std::string,
    std::tuple<std::tuple<int64_t, int64_t>, std::tuple<int64_t, int64_t>>>;

/// Transforms the given codes from one precision to another. If the given
/// precision is higher than the precision of the given codes, the result
/// contains a zoom in, otherwise it contains a zoom out.
[[nodiscard]] auto transform(const pybind11::array &hash, uint32_t precision)
    -> pybind11::array;

}  // namespace pyinterp::geohash::string
