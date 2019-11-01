// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/detail/axis/container.hpp"

namespace pyinterp::detail::axis::container {

void Irregular::make_edges() {
  auto n = points_.size();
  edges_.resize(n + 1);

  for (Eigen::Index ix = 1; ix < n; ++ix) {
    edges_[ix] = (points_[ix - 1] + points_[ix]) / 2;
  }

  edges_[0] = 2 * points_[0] - edges_[1];
  edges_[n] = 2 * points_[n - 1] - edges_[n - 1];
}

Irregular::Irregular(Eigen::VectorXd points) : points_(std::move(points)) {
  if (points_.size() == 0) {
    throw std::invalid_argument("unable to create an empty container.");
  }
  is_ascending_ = calculate_is_ascending();
  make_edges();
}

auto Irregular::flip() -> void {
  std::reverse(points_.data(), points_.data() + points_.size());
  is_ascending_ = !is_ascending_;
  make_edges();
}

auto Irregular::find_index(double coordinate, bool bounded) const -> int64_t {
  int64_t low = 0;
  int64_t mid = 0;
  int64_t high = size();

  if (is_ascending_) {
    if (coordinate < edges_[0]) {
      return bounded ? 0 : -1;
    }

    if (coordinate > edges_[edges_.size() - 1]) {
      return bounded ? high - 1 : -1;
    }

    while (high > low + 1) {
      mid = (low + high) >> 1;  // NOLINT (low and high are strictly positive)
      auto value = edges_[mid];

      if (value == coordinate) {
        return mid;
      }
      value < coordinate ? low = mid : high = mid;
    }
    return low;
  }

  if (coordinate < edges_[edges_.size() - 1]) {
    return bounded ? high - 1 : -1;
  }

  if (coordinate > edges_[0]) {
    return bounded ? 0 : -1;
  }

  while (high > low + 1) {
    mid = (low + high) >> 1;  // NOLINT (low and high are strictly positive)
    auto value = edges_[mid];

    if (value == coordinate) {
      return mid;
    }
    value < coordinate ? high = mid : low = mid;
  }
  return low;
}

}  // namespace pyinterp::detail::axis::container
