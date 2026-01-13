// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/detail/comparable_distance/interface.hpp>
#include <numbers>
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

  /// @brief Finds the nearest vertices in both linestrings to a given point
  /// using the golden-section search algorithm.
  /// @param[in] point The point to which the nearest vertices are sought
  /// @return A tuple containing the indices of the nearest vertices
  /// in both linestrings
  [[nodiscard]] auto nearest(const Point& point) const
      -> std::tuple<size_t, size_t> {
    auto p1 = Crossover::nearest_vertex_golden(point, line1_);
    auto p2 = Crossover::nearest_vertex_golden(point, line2_);
    return {p1.first, p2.first};
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
  /// point using the golden-section search algorithm.
  /// @tparam Point Type of the query point
  /// @tparam LineString Type of the linestring
  /// @param[in] query The point to which the nearest vertex is sought
  /// @param[in] line The linestring containing the vertices
  /// @return A pair containing the index of the nearest vertex and the distance
  /// to it
  static auto nearest_vertex_golden(Point const& query,
                                    LineString<Point> const& line)
      -> std::pair<size_t, typename boost::geometry::default_distance_result<
                               Point, Point>::type> {
    size_t lo = 0;
    size_t hi = line.size() - 1;

    auto dist = [&](size_t i) {
      return boost::geometry::comparable_distance(query, line[i]);
    };

    auto m1 = hi - static_cast<size_t>((hi - lo) / std::numbers::phi);
    auto m2 = lo + static_cast<size_t>((hi - lo) / std::numbers::phi);
    auto d1 = dist(m1);
    auto d2 = dist(m2);

    while (hi - lo > 2) {
      if (d1 < d2) {
        hi = m2;
        m2 = m1;
        d2 = d1;
        m1 = hi - static_cast<std::size_t>((hi - lo) / std::numbers::phi);
        d1 = dist(m1);
      } else {
        lo = m1;
        m1 = m2;
        d1 = d2;
        m2 = lo + static_cast<std::size_t>((hi - lo) / std::numbers::phi);
        d2 = dist(m2);
      }
    }

    // Final linear scan
    std::size_t best_idx = lo;
    auto best_dist = dist(lo);
    for (std::size_t i = lo + 1; i <= hi; ++i) {
      if (auto d = dist(i); d < best_dist) {
        best_dist = d;
        best_idx = i;
      }
    }

    return {best_idx, boost::geometry::distance(query, line[best_idx])};
  }
};

}  // namespace pyinterp::geometry
