// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/geodetic/box.hpp"
#include "pyinterp/geodetic/point.hpp"

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
            [](void* ptr) { delete static_cast<std::vector<char>*>(ptr); }),
        chars_(precision),
        size_(size) {}

  /// Get the pointer to the raw memory
  [[nodiscard]] inline auto buffer() const -> char* { return array_->data(); }

  /// Creates the numpy array from the memory allocated in the C++ code without
  /// copying the data.
  [[nodiscard]] inline auto pyarray() -> pybind11::array {
    return pybind11::array(pybind11::dtype("S" + std::to_string(chars_)),
                           {size_}, {chars_ * sizeof(char)}, array_->data(),
                           capsule_);
  }

  static auto get_info(const pybind11::array& hash, ssize_t ndim)
      -> pybind11::buffer_info;

 private:
  std::vector<char>* array_;
  pybind11::capsule capsule_;
  uint32_t chars_;
  size_t size_;
};

/// Encode a point into geohash with the given bit depth
auto encode(const geodetic::Point& point, char* buffer, uint32_t precision)
    -> void;

/// Encode points into geohash with the given bit depth
[[nodiscard]] auto encode(const Eigen::Ref<const Eigen::VectorXd>& lon,
                          const Eigen::Ref<const Eigen::VectorXd>& lat,
                          uint32_t precision) -> pybind11::array;

/// Returns the region encoded
[[nodiscard]] auto bounding_box(const char* hash, size_t count)
    -> geodetic::Box;

/// Decode a hash into a spherical equatorial point. If round is true, the
/// coordinates of the points will be rounded to the accuracy defined by the
/// GeoHash.
[[nodiscard]] auto decode(const char* hash, size_t count, bool round)
    -> geodetic::Point;

/// Decode hashes into a spherical equatorial points. If round is true, the
/// coordinates of the points will be rounded to the accuracy defined by the
/// GeoHash.
[[nodiscard]] auto decode(const pybind11::array& hash, bool round)
    -> std::tuple<Eigen::VectorXd, Eigen::VectorXd>;

/// Returns all neighbors hash clockwise from north around northwest at the
/// given precision:
///   7 0 1
///   6 x 2
///   5 4 3
[[nodiscard]] auto neighbors(const char* hash, size_t count) -> pybind11::array;

/// Returns the area covered by the GeoHash
[[nodiscard]] inline auto area(const char* const hash, size_t count,
                               const std::optional<geodetic::System>& wgs)
    -> double {
  return bounding_box(hash, count).area(wgs);
}

/// Returns the area covered by the GeoHash codes
[[nodiscard]] auto area(const pybind11::array& hash,
                        const std::optional<geodetic::System>& wgs)
    -> Eigen::MatrixXd;

/// Returns all GeoHash with the defined box
[[nodiscard]] auto bounding_boxes(const std::optional<geodetic::Box>& box,
                                  uint32_t precision) -> pybind11::array;

/// Returns the start and end indexes of the different GeoHash boxes.
[[nodiscard]] auto where(const pybind11::array& hash)
    -> std::map<std::string, std::tuple<std::tuple<int64_t, int64_t>,
                                        std::tuple<int64_t, int64_t>>>;

}  // namespace pyinterp::geohash::string