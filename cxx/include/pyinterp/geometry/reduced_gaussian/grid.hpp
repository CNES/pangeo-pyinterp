// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/reduced_gaussian/mesh.hpp"
#include "pyinterp/geometry/reduced_gaussian/zone.hpp"

namespace pyinterp::geometry::reduced_gaussian {

/// @brief Type of the values of a field.
enum class DType : uint8_t {
  kFloat32,  ///< `float`
  kFloat64,  ///< `double`
};

/// @brief A non-owning view of the values of one field.
///
/// The values are row-major: one row per point, `width` values per row (the
/// model levels of a 3D field, 1 for a 2D field), so that the column of a
/// point is contiguous.
struct Field {
  /// @brief Name of the field.
  std::string name;
  /// @brief First value.
  const void* data{nullptr};
  /// @brief Number of points.
  int64_t rows{0};
  /// @brief Number of values per point.
  int64_t width{1};
  /// @brief Type of the values.
  DType dtype{DType::kFloat64};
  /// @brief Whether the field was given with a level dimension, which the
  ///        results then keep even for a single level.
  bool has_levels{false};
};

/// @brief The fields of one time step on a mesh, or on a zone of it.
///
/// The grid does not own its values: the caller keeps them alive for the
/// lifetime of the grid.
class Grid {
 public:
  /// @brief Build the grid.
  /// @param[in] mesh The mesh.
  /// @param[in] zone The points held, or null for a global grid.
  /// @param[in] fields The fields, each with one row per point held.
  /// @param[in] a,b Hybrid coefficients of the half levels, or empty.
  /// @throws std::invalid_argument on inconsistent inputs.
  Grid(std::shared_ptr<const Mesh> mesh, std::shared_ptr<const Zone> zone,
       std::vector<Field> fields, Vector<double> a = {}, Vector<double> b = {})
      : mesh_(std::move(mesh)),
        zone_(std::move(zone)),
        fields_(std::move(fields)),
        a_(std::move(a)),
        b_(std::move(b)) {
    if (!mesh_) {
      throw std::invalid_argument("the grid needs a mesh");
    }
    if (zone_ && zone_->mesh_size() != mesh_->size()) {
      throw std::invalid_argument("the zone was built for a mesh of " +
                                  std::to_string(zone_->mesh_size()) +
                                  " points, not " +
                                  std::to_string(mesh_->size()));
    }
    if (fields_.empty()) {
      throw std::invalid_argument("the grid needs at least one field");
    }
    const auto rows = zone_ ? zone_->size() : mesh_->size();
    for (size_t ix = 0; ix < fields_.size(); ++ix) {
      const auto& field = fields_[ix];
      if (field.rows != rows) {
        throw std::invalid_argument(
            "field '" + field.name + "' has " + std::to_string(field.rows) +
            " points, expected " + std::to_string(rows) +
            (zone_ ? " (the size of the zone)" : " (the size of the mesh)"));
      }
      if (field.width < 1 || field.data == nullptr) {
        throw std::invalid_argument("field '" + field.name + "' is empty");
      }
      for (size_t jx = 0; jx < ix; ++jx) {
        if (fields_[jx].name == field.name) {
          throw std::invalid_argument("duplicate field '" + field.name + "'");
        }
      }
    }
    if (a_.size() != b_.size()) {
      throw std::invalid_argument(
          "the hybrid coefficients a and b must have the same size");
    }
  }

  /// @brief The mesh.
  [[nodiscard]] auto mesh() const noexcept -> const Mesh& { return *mesh_; }

  /// @brief The mesh, shared.
  [[nodiscard]] auto mesh_ptr() const noexcept
      -> const std::shared_ptr<const Mesh>& {
    return mesh_;
  }

  /// @brief The zone, or null for a global grid.
  [[nodiscard]] auto zone() const noexcept
      -> const std::shared_ptr<const Zone>& {
    return zone_;
  }

  /// @brief The fields.
  [[nodiscard]] auto fields() const noexcept -> const std::vector<Field>& {
    return fields_;
  }

  /// @brief Hybrid coefficient A of the half levels (empty if not given).
  [[nodiscard]] auto a() const noexcept -> const Vector<double>& { return a_; }

  /// @brief Hybrid coefficient B of the half levels (empty if not given).
  [[nodiscard]] auto b() const noexcept -> const Vector<double>& { return b_; }

  /// @brief The row of a point of the mesh in the values.
  /// @return The row, or -1 if the grid does not hold the point.
  [[nodiscard]] auto row(int64_t index) const noexcept -> int64_t {
    return zone_ ? zone_->row(index) : index;
  }

  /// @brief Whether another grid has the same fields (names and widths, in
  ///        the same order).
  [[nodiscard]] auto same_fields(const Grid& other) const noexcept -> bool {
    return std::ranges::equal(
        fields_, other.fields_, [](const Field& lhs, const Field& rhs) {
          return lhs.name == rhs.name && lhs.width == rhs.width;
        });
  }

 private:
  std::shared_ptr<const Mesh> mesh_;
  std::shared_ptr<const Zone> zone_;
  std::vector<Field> fields_;
  Vector<double> a_;
  Vector<double> b_;
};

/// @brief The time steps that bracket a date, and the weight of the later.
struct Bracket {
  /// @brief The earlier step.
  int64_t i0{0};
  /// @brief The later step; equal to `i0` when a single step is read.
  int64_t i1{0};
  /// @brief Weight of `i1`, in [0, 1].
  double weight{0};

  /// @brief Whether a single step is read.
  [[nodiscard]] constexpr auto single() const noexcept -> bool {
    return i0 == i1;
  }
};

/// @brief How the steps of a date are chosen.
struct BracketRule {
  /// @brief Take the nearest step instead of blending the two.
  bool nearest{false};
  /// @brief Width of the analysis intervals, or 0 for the two steps that
  ///        bracket the date.
  ///
  /// With an interval, a date is read between the two ends of the interval
  /// it falls in, on a lattice whose phase is the first step: the pair is
  /// then a statement about the series (these two analyses), not whatever
  /// pair happens to bracket the date. An interval that lost one of its
  /// bounds is read off the other alone; one that lost both reads nothing.
  int64_t interval{0};
  /// @brief Largest time between the two steps of a pair, or 0 for no
  ///        limit; a wider pair is a missing step.
  int64_t max_step_gap{0};
};

/// @brief A time series of grids on one mesh.
///
/// Every grid shares the mesh: a point is located once and its cell read in
/// each step. The zones may differ from step to step, each built from the
/// points that read the step.
class Series {
 public:
  /// @brief Build an empty series.
  explicit Series(std::shared_ptr<const Mesh> mesh) : mesh_(std::move(mesh)) {
    if (!mesh_) {
      throw std::invalid_argument("the series needs a mesh");
    }
  }

  /// @brief Add a step.
  /// @param[in] time Date of the step.
  /// @param[in] grid The grid.
  /// @throws std::invalid_argument if the grid is on another mesh, has other
  ///         fields or hybrid coefficients than the steps already held, or if
  ///         the series already has a step at this date.
  void append(int64_t time, std::shared_ptr<const Grid> grid) {
    if (!grid) {
      throw std::invalid_argument("the grid is null");
    }
    if (grid->mesh_ptr() != mesh_ && !(grid->mesh() == *mesh_)) {
      throw std::invalid_argument("the grid is on another mesh");
    }
    if (!grids_.empty()) {
      const auto& first = *grids_.front();
      if (!first.same_fields(*grid)) {
        throw std::invalid_argument(
            "the grid has other fields than the series (same names and "
            "widths, in the same order, are required)");
      }
      if (first.a() != grid->a() || first.b() != grid->b()) {
        throw std::invalid_argument(
            "the grid has other hybrid coefficients than the series");
      }
    }
    const auto it = std::ranges::lower_bound(times_, time);
    if (it != times_.end() && *it == time) {
      throw std::invalid_argument("the series already has a step at " +
                                  std::to_string(time));
    }
    const auto pos = it - times_.begin();
    times_.insert(it, time);
    grids_.insert(grids_.begin() + pos, std::move(grid));
  }

  /// @brief Remove the steps strictly before a date.
  void drop_before(int64_t time) {
    const auto count = std::ranges::lower_bound(times_, time) - times_.begin();
    times_.erase(times_.begin(), times_.begin() + count);
    grids_.erase(grids_.begin(), grids_.begin() + count);
  }

  /// @brief The mesh.
  [[nodiscard]] auto mesh() const noexcept -> const Mesh& { return *mesh_; }

  /// @brief Number of steps.
  [[nodiscard]] auto size() const noexcept -> int64_t {
    return static_cast<int64_t>(times_.size());
  }

  /// @brief Dates of the steps, sorted.
  [[nodiscard]] auto times() const noexcept -> const std::vector<int64_t>& {
    return times_;
  }

  /// @brief The grid of a step.
  [[nodiscard]] auto grid(int64_t ix) const -> const Grid& {
    return *grids_.at(static_cast<size_t>(ix));
  }

  /// @brief The steps a date reads.
  /// @return The steps, or nothing if the date is outside the series or a
  ///         step it needs is missing.
  [[nodiscard]] auto bracket(int64_t time, const BracketRule& rule) const
      -> std::optional<Bracket> {
    return bracket(times_, time, rule);
  }

  /// @brief The steps a date reads among the dates of a series.
  ///
  /// The rule of `bracket`, over the dates alone, so that the steps a date
  /// will read can be known before any grid is: to build the zone of each
  /// step, or to report which steps are missing.
  ///
  /// @param[in] times Dates of the steps, sorted and distinct.
  /// @param[in] time The date.
  /// @param[in] rule How the steps of a date are chosen.
  /// @return The steps, as indices in @p times, or nothing.
  [[nodiscard]] static auto bracket(const std::vector<int64_t>& times,
                                    int64_t time, const BracketRule& rule)
      -> std::optional<Bracket> {
    if (times.empty()) {
      return std::nullopt;
    }
    auto result = rule.interval > 0 ? interval_pair(times, time, rule)
                                    : nearest_pair(times, time);
    if (!result) {
      return std::nullopt;
    }
    if (!result->single() && rule.max_step_gap > 0 &&
        times[result->i1] - times[result->i0] > rule.max_step_gap) {
      return std::nullopt;
    }
    if (rule.nearest && !result->single()) {
      const auto chosen = result->weight <= 0.5 ? result->i0 : result->i1;
      result = Bracket{.i0 = chosen, .i1 = chosen, .weight = 0.0};
    }
    return result;
  }

 private:
  std::shared_ptr<const Mesh> mesh_;
  std::vector<int64_t> times_;
  std::vector<std::shared_ptr<const Grid>> grids_;

  /// @brief The index of a date among the steps, if a step has it.
  [[nodiscard]] static auto find(const std::vector<int64_t>& times,
                                 int64_t time) -> std::optional<int64_t> {
    const auto it = std::ranges::lower_bound(times, time);
    if (it == times.end() || *it != time) {
      return std::nullopt;
    }
    return it - times.begin();
  }

  /// @brief The two steps that bracket a date.
  [[nodiscard]] static auto nearest_pair(const std::vector<int64_t>& times,
                                         int64_t time)
      -> std::optional<Bracket> {
    if (time < times.front() || time > times.back()) {
      return std::nullopt;
    }
    const auto it = std::ranges::upper_bound(times, time);
    const auto i1 = it - times.begin();
    const auto i0 = i1 - 1;
    if (times[i0] == time || std::cmp_equal(i1, times.size())) {
      return Bracket{.i0 = i0, .i1 = i0, .weight = 0.0};
    }
    return Bracket{.i0 = i0,
                   .i1 = i1,
                   .weight = static_cast<double>(time - times[i0]) /
                             static_cast<double>(times[i1] - times[i0])};
  }

  /// @brief The two ends of the interval a date falls in.
  [[nodiscard]] static auto interval_pair(const std::vector<int64_t>& times,
                                          int64_t time, const BracketRule& rule)
      -> std::optional<Bracket> {
    const auto origin = times.front();
    const auto offset = time - origin;
    // Floor division: a date before the first step falls in an interval that
    // the series does not hold.
    auto k = offset / rule.interval;
    if (offset % rule.interval != 0 && offset < 0) {
      --k;
    }
    const auto start = origin + (k * rule.interval);
    const auto i0 = find(times, start);
    if (time == start && i0) {
      return Bracket{.i0 = *i0, .i1 = *i0, .weight = 0.0};
    }
    const auto i1 = find(times, start + rule.interval);
    // A bound that could not be fetched leaves the date to the other one,
    // held as it stands rather than interpolated across the hole.
    if (!i0 && !i1) {
      return std::nullopt;
    }
    if (!i0 || !i1) {
      const auto alone = i0 ? *i0 : *i1;
      return Bracket{.i0 = alone, .i1 = alone, .weight = 0.0};
    }
    return Bracket{.i0 = *i0,
                   .i1 = *i1,
                   .weight = static_cast<double>(time - start) /
                             static_cast<double>(rule.interval)};
  }
};

}  // namespace pyinterp::geometry::reduced_gaussian
