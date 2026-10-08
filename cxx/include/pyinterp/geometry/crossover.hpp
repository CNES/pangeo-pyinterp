// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/detail/comparable_distance/interface.hpp>
#include <stdexcept>
#include <vector>

#include "pyinterp/geometry/linestring.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::geometry {

/// @brief Class to find and get properties of crossover points between two
/// linestrings.
/// @tparam Point Type of point
template <typename Point>
class Crossover {
 public:
  /// @brief Constructs a crossover object from two linestrings
  /// @param[in] line1 First linestring
  /// @param[in] line2 Second linestring
  Crossover(LineString<Point> line1, LineString<Point> line2)
      : line1_(std::move(line1)), line2_(std::move(line2)) {}

  /// @brief Get the first linestring
  /// @return First linestring
  [[nodiscard]]
  constexpr auto line1() const noexcept -> const LineString<Point>& {
    return line1_;
  }

  /// @brief Get the second linestring
  /// @return Second linestring
  [[nodiscard]]
  constexpr auto line2() const noexcept -> const LineString<Point>& {
    return line2_;
  }

  /// @brief Finds the nearest vertices in both linestrings to a given point.
  /// @param[in] point The point to which the nearest vertices are sought
  /// @param[in] assume_unimodal If true, the distance from the point to the
  /// vertices is assumed to be unimodal along each linestring, and the nearest
  /// vertices are found with a bisection search; otherwise, all the vertices
  /// are examined.
  /// @return A tuple containing the indices of the nearest vertices
  /// in both linestrings
  [[nodiscard]] auto nearest(const Point& point,
                             const bool assume_unimodal = false) const
      -> std::tuple<size_t, size_t> {
    if (assume_unimodal) {
      return {Crossover::nearest_vertex_bisection(point, line1_),
              Crossover::nearest_vertex_bisection(point, line2_)};
    }
    return {Crossover::nearest_vertex_linear(point, line1_),
            Crossover::nearest_vertex_linear(point, line2_)};
  }

  /// @brief Serialize the Crossover state for storage or transmission.
  /// @return Serialized state as a Writer object
  [[nodiscard]] constexpr auto pack() const -> serialization::Writer {
    serialization::Writer writer;
    writer.write(kMagicNumber);
    writer.write(line1_.pack());
    writer.write(line2_.pack());
    return writer;
  }

  /// @brief Deserialize a Crossover from serialized state.
  /// @param[in] state Reference to serialization Reader containing encoded
  /// Crossover data
  /// @return New Crossover instance with restored properties
  [[nodiscard]] static auto unpack(serialization::Reader& state)
      -> Crossover<Point> {
    auto magic_number = state.read<uint32_t>();
    if (magic_number != kMagicNumber) {
      throw std::runtime_error("Invalid magic number for Crossover");
    }
    auto unpack_ls = [](serialization::Reader& state) {
      auto ls_state = state.read_vector<std::byte>();
      auto reader = serialization::Reader(std::move(ls_state));
      return LineString<Point>::unpack(reader);
    };
    auto line1 = unpack_ls(state);
    auto line2 = unpack_ls(state);
    return Crossover(std::move(line1), std::move(line2));
  }

 protected:
  /// First linestring
  LineString<Point> line1_;
  /// Second linestring
  LineString<Point> line2_;

 private:
  /// Magic number for Crossover serialization
  static constexpr uint32_t kMagicNumber = 0x5F585F5F;  // "_X__"

  /// @brief Finds the index of the nearest vertex in a linestring to a given
  /// point by examining all the vertices.
  /// @param[in] query The point to which the nearest vertex is sought
  /// @param[in] line The linestring containing the vertices
  /// @return The index of the nearest vertex
  static auto nearest_vertex_linear(Point const& query,
                                    LineString<Point> const& line) -> size_t {
    size_t best_idx = 0;
    auto best_dist = boost::geometry::comparable_distance(query, line[0]);
    for (size_t ix = 1; ix < line.size(); ++ix) {
      if (auto dist = boost::geometry::comparable_distance(query, line[ix]);
          dist < best_dist) {
        best_dist = dist;
        best_idx = ix;
      }
    }
    return best_idx;
  }

  /// @brief Finds the index of the nearest vertex in a linestring to a given
  /// point using a bisection on the sign of the distance variation between
  /// consecutive vertices.
  /// @param[in] query The point to which the nearest vertex is sought
  /// @param[in] line The linestring containing the vertices
  /// @return The index of the nearest vertex
  /// @note The distance from the query point to the vertices must be strictly
  /// unimodal along the linestring: the result is undefined if the linestring
  /// contains duplicated vertices or comes back towards the query point.
  static auto nearest_vertex_bisection(Point const& query,
                                       LineString<Point> const& line)
      -> size_t {
    size_t lo = 0;
    size_t hi = line.size() - 1;

    auto dist = [&](size_t i) {
      return boost::geometry::comparable_distance(query, line[i]);
    };

    // Invariant: the nearest vertex lies in [lo, hi]. The distance decreases
    // before the nearest vertex and increases after it.
    while (lo < hi) {
      const auto mid = lo + (hi - lo) / 2;
      if (dist(mid) <= dist(mid + 1)) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }

    return lo;
  }
};

}  // namespace pyinterp::geometry
