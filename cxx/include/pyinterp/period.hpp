// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "pyinterp/eigen.hpp"

namespace pyinterp {

/// @brief Represents a half-open time period [begin, end).
struct Period {
  int64_t begin{0};  ///< Start of the period (inclusive).
  int64_t last{-1};  ///< Last value in the period (inclusive).

  /// @brief Create a null/default period.
  constexpr Period() = default;

  /// @brief Create a period from begin to end.
  /// @param[in] begin Start of the period (inclusive).
  /// @param[in] end End of the period.
  /// @param[in] within If true, 'end' is treated as the last inclusive value
  /// [begin, end]. If false, 'end' is treated as exclusive [begin, end).
  constexpr Period(int64_t begin, int64_t end, bool within = true) noexcept
      : begin(begin), last(end - (within ? 0 : 1)) {}

  /// @brief Return one past the last element (exclusive end).
  [[nodiscard]] constexpr auto end() const noexcept -> int64_t {
    return last + 1;
  }

  /// @brief Return the duration.
  [[nodiscard]] constexpr auto duration() const noexcept -> int64_t {
    return end() - begin;
  }

  /// @brief True if period is ill-formed (length is zero or negative).
  [[nodiscard]] constexpr auto is_null() const noexcept -> bool {
    return last < begin;
  }

  /// @brief Default equality and ordering (lexicographic on begin, then last).
  [[nodiscard]] constexpr auto operator<=>(const Period&) const noexcept =
      default;

  /// @brief True if the point is inside the period [begin, last].
  [[nodiscard]] constexpr auto contains(int64_t point) const noexcept -> bool {
    return (point >= begin) && (point <= last);
  }

  /// @brief True if this period fully contains (or equals) the other period.
  [[nodiscard]] constexpr auto contains(const Period& other) const noexcept
      -> bool {
    return (begin <= other.begin) && (last >= other.last);
  }

  /// @brief True if periods are adjacent without a gap.
  [[nodiscard]] constexpr auto is_adjacent(const Period& other) const noexcept
      -> bool {
    return (other.begin == end()) || (begin == other.end());
  }

  /// @brief True if the entire period is after the given point (point < begin).
  [[nodiscard]] constexpr auto is_after(int64_t point) const noexcept -> bool {
    return !is_null() && (point < begin);
  }

  /// @brief True if the entire period is before the given point (last < point).
  [[nodiscard]] constexpr auto is_before(int64_t point) const noexcept -> bool {
    return !is_null() && (last < point);
  }

  /// @brief True if the periods overlap in any way.
  [[nodiscard]] constexpr auto intersects(const Period& other) const noexcept
      -> bool {
    return (begin <= other.last) && (other.begin <= last);
  }

  /// @brief True if the given date is within tolerance of the period.
  /// @param date The date to check.
  /// @param tolerance The tolerance margin (inclusive).
  [[nodiscard]] constexpr auto is_close(int64_t date,
                                        int64_t tolerance) const noexcept
      -> bool {
    return (date >= begin - tolerance) && (date <= last + tolerance);
  }

  /// @brief Return the intersection of two periods.
  /// @return A new Period representing the intersection. If disjoint, returns
  /// a null period.
  [[nodiscard]] constexpr auto intersection(const Period& other) const noexcept
      -> Period {
    if (!intersects(other)) {
      return Period{};
    }
    return Period{std::max(begin, other.begin), std::min(last, other.last)};
  }

  /// @brief Returns the union of intersecting or adjacent periods.
  /// @return A new Period representing the union. If disjoint (and not
  /// adjacent), returns a null period.
  [[nodiscard]] constexpr auto merge(const Period& other) const noexcept
      -> Period {
    if (!intersects(other) && !is_adjacent(other)) {
      return Period{};
    }
    return Period{std::min(begin, other.begin), std::max(last, other.last)};
  }

  /// @brief Extend the period to include the given point.
  [[nodiscard]] constexpr auto extend(int64_t point) const noexcept -> Period {
    return Period{std::min(begin, point), std::max(last, point)};
  }

  /// @brief Shift the period by an offset.
  [[nodiscard]] constexpr auto shift(int64_t offset) const noexcept -> Period {
    return Period{begin + offset, last + offset};
  }
};

/// @brief A list of periods stored in a std::vector.
///
/// All periods in the list share the same resolution, managed at the
/// container level. This avoids redundant storage and simplifies operations.
class PeriodList : public std::vector<Period> {
 public:
  using std::vector<Period>::vector;

  /// @brief Create a PeriodList from a vector of periods.
  /// @param[in,out] periods The vector of periods.
  explicit constexpr PeriodList(std::vector<Period>&& periods) noexcept
      : std::vector<Period>(std::move(periods)) {}

  /// @brief Check that periods are sorted and non-overlapping.
  [[nodiscard]] auto is_sorted_and_disjoint() const noexcept -> bool {
    if (size() <= 1) {
      return true;
    }
    // Check if any adjacent periods intersect or are out of order
    return std::ranges::adjacent_find(
               *this, [](const Period& a, const Period& b) {
                 return a.intersects(b) || b.begin < a.begin;
               }) == end();
  }

  /// @brief Sort periods by begin time.
  auto sort() -> void {
    std::ranges::sort(*this, {}, [](const Period& p) { return p.begin; });
  }

  /// @brief Merge two lists of periods.
  /// @param[in] other The other PeriodList to merge with.
  /// @return A new PeriodList with merged periods.
  inline auto merge(const PeriodList& other) -> void;

  /// @brief Check if a date is within tolerance of any period.
  [[nodiscard]] auto is_close(int64_t date, int64_t tolerance) const noexcept
      -> bool {
    return std::ranges::any_of(
        *this, [=](const auto& p) { return p.is_close(date, tolerance); });
  }

  /// @brief Get the duration covered by all periods.
  /// @note This algorithm assumes that periods are sorted.
  /// @return The total duration.
  [[nodiscard]] auto duration() const noexcept -> int64_t {
    if (empty()) {
      return 0;
    }

    if (size() == 1) {
      return front().duration();
    }

    return back().end() - front().begin;
  }

  /// @brief Get the aggregate duration of all periods.
  /// @note This algorithm assumes that periods are sorted and disjoint.
  /// @return The aggregate duration.
  [[nodiscard]] inline auto aggregate_duration() const noexcept -> int64_t;

  /// @brief Find the period containing a date using binary search.
  /// @param date The date (must be in list's resolution)
  /// @return Pointer to the containing period, or nullptr if not found
  [[nodiscard]] inline auto find_containing(int64_t date) const noexcept
      -> const Period*;

  /// @brief Find the index of the period containing a date.
  /// @param date The date (must be in list's resolution)
  /// @return Index of the containing period, or -1 if not found
  [[nodiscard]] inline auto find_containing_index(int64_t date) const noexcept
      -> int64_t {
    if (const auto* p = find_containing(date)) {
      return std::distance(data(), p);
    }
    return -1;
  }

  /// @brief Filter to only periods within the given period.
  /// @param period The period to filter against.
  /// @return A new PeriodList with periods fully contained in 'period'.
  [[nodiscard]] inline auto filter_contained(
      const Period& period) const noexcept -> PeriodList {
    PeriodList result;
    result.reserve(size());
    for (const auto& p : *this) {
      if (period.contains(p)) {
        result.push_back(p);
      }
    }
    result.shrink_to_fit();
    return result;
  }

  /// @brief Filter to only periods longer than the given duration.
  /// @param min_duration The minimum duration.
  /// @return A new PeriodList with periods longer than 'min_duration'.
  [[nodiscard]] inline auto filter_min_duration(
      int64_t min_duration) const noexcept -> PeriodList {
    PeriodList result;
    result.reserve(size());
    for (const auto& p : *this) {
      if (p.duration() >= min_duration) {
        result.push_back(p);
      }
    }
    result.shrink_to_fit();
    return result;
  }

  /// @brief Identify dates that cross (enter or are within) any managed period.
  ///
  /// For each input date, determines whether it either:
  /// - Falls inside one of the periods, OR
  /// - Has at least one period beginning at or after it (before the last input
  ///   date)
  ///
  /// This effectively identifies dates that "encounter" a period in the
  /// temporal sequence, excluding only those dates that come after all periods
  /// have ended.
  ///
  /// @param dates Vector of dates to check (should be sorted for optimal
  /// performance).
  /// @return A vector of booleans where true indicates the date crosses a
  /// period.
  ///
  /// @note If the last date in the input falls within a period, all dates are
  /// marked as true since they all precede or coincide with a period.
  [[nodiscard]] inline auto cross_a_period(
      const Eigen::Ref<const Vector<int64_t>>& dates) const -> Vector<bool>;

  /// @brief Test which dates fall within any of the managed periods.
  ///
  /// Efficiently checks each input date to determine if it is contained within
  /// at least one period in the list. Uses a single-pass algorithm that
  /// advances through the periods as dates are processed.
  ///
  /// @param dates Vector of dates to check (should be sorted for optimal
  /// performance).
  /// @return Boolean vector where true indicates the date is contained in a
  /// period.
  ///
  /// @note This is a simple membership test - returns true only if the date
  /// falls within [begin, last] of some period. Compare with cross_a_period()
  /// which has more complex "look-ahead" semantics.
  [[nodiscard]] inline auto belong_to_a_period(
      const Eigen::Ref<const Vector<int64_t>>& dates) const -> Vector<bool>;

  /// @brief Join adjacent periods together.
  /// @param epsilon The maximum gap between periods to join.
  /// @return A new PeriodList with adjacent periods joined.
  [[nodiscard]] inline auto join_adjacent_periods(int64_t epsilon) const
      -> PeriodList;
};

// ============================================================================
// Implementation
// ============================================================================

auto PeriodList::merge(const PeriodList& other) -> void {
  auto periods = PeriodList();
  periods.reserve(size() + other.size());

  size_t ix = 0;
  size_t jx = 0;

  auto insert_or_merge = [&periods](const Period& period) -> void {
    if (periods.empty()) {
      periods.push_back(period);
      return;
    }
    auto& last_period = periods.back();
    auto overlap = last_period.merge(period);
    if (overlap.is_null()) {
      periods.push_back(period);
    } else {
      last_period = overlap;
    }
  };

  while (ix < size() && jx < other.size()) {
    insert_or_merge(other[jx].is_after((*this)[ix].begin) ? (*this)[ix++]
                                                          : other[jx++]);
  }

  while (ix < size()) {
    insert_or_merge((*this)[ix++]);
  }

  while (jx < other.size()) {
    insert_or_merge(other[jx++]);
  }

  periods.shrink_to_fit();
  *this = std::move(periods);
}

// ============================================================================

auto PeriodList::aggregate_duration() const noexcept -> int64_t {
  if (empty()) {
    return 0;
  }

  int64_t total = 0;
  auto current_begin = front().begin;
  auto current_end = front().end();

  for (size_t i = 1; i < size(); ++i) {
    const auto& p = (*this)[i];
    if (p.begin <= current_end) {
      // Overlapping or adjacent: extend coverage
      current_end = std::max(current_end, p.end());
    } else {
      // Gap: accumulate current segment and start new
      total += (current_end - current_begin);
      current_begin = p.begin;
      current_end = p.end();
    }
  }

  // Don't forget to add the last segment
  total += (current_end - current_begin);

  return total;
}

// ============================================================================

auto PeriodList::find_containing(int64_t date) const noexcept -> const Period* {
  auto it = std::ranges::lower_bound(*this, date, {},
                                     [](const Period& p) { return p.begin; });

  // Check the previous period (date might be inside it, since lower_bound
  // returns first element >= date)
  if (it != begin()) {
    if (auto prev = std::prev(it); prev->contains(date)) {
      return &(*prev);
    }
  }

  // Check the current period (if date == begin)
  if (it != end() && it->contains(date)) {
    return &(*it);
  }

  return nullptr;
}

// ============================================================================

auto PeriodList::cross_a_period(
    const Eigen::Ref<const Vector<int64_t>>& dates) const -> Vector<bool> {
  // The index of the closest period of the current date processed.
  int64_t first_index = 0;

  // Flag equal to 1 if the date belongs to a period, 0 otherwise.
  auto flags = Vector<bool>(dates.size());
  flags.setConstant(false);

  // Index of the traversed date.
  int64_t ix = 0;
  // The last date processed.
  const auto last_date = dates[dates.size() - 1];

  // Searches for the period containing or after the provided date.
  auto lookup = [this](
                    const int64_t first_index,
                    const int64_t date) -> std::tuple<int64_t, const Period*> {
    for (auto index = first_index; std::cmp_less(index, size()); ++index) {
      const auto* period = &(*this)[index];
      if (period->contains(date) || period->is_after(date)) {
        return {index, period};
      }
    }
    return {-1, nullptr};
  };

  // The index of the first period that is located after the last date
  // provided.
  auto [last_index, period] = lookup(0, last_date);
  if (period != nullptr && period->contains(last_date)) {
    // If the last date processed belongs to a period, no other date can be
    // outside the periods.
    flags.setConstant(true);
    return flags;
  }

  while (ix < dates.size()) {
    const auto date = dates(ix);

    std::tie(first_index, period) = lookup(first_index, date);
    if (period == nullptr || first_index == last_index) {
      // If the date is not in any period, or if the period is the last one
      // after the last date processed, the inspection is over.
      break;
    }

    // If the date belongs to a period or if there is a period after this date
    // which is not the first period after the last supplied date, then a
    // period is traversed.
    if (period->contains(date) || period->is_after(date)) {
      flags(ix) = true;
    }
    // Move to the next date.
    ++ix;
  }
  return flags;
}

// ============================================================================

auto PeriodList::belong_to_a_period(
    const Eigen::Ref<const Vector<int64_t>>& dates) const -> Vector<bool> {
  // Flag equal to 1 if the date belongs to a period, 0 otherwise.
  auto flags = Vector<bool>(dates.size());
  flags.setConstant(false);

  // Index of the traversed date.
  int64_t ix = 0;
  auto it = begin();
  auto it_end = end();

  while (ix < dates.size()) {
    const auto date = dates(ix);
    while (it != it_end && !it->contains(date) && !it->is_after(date)) {
      ++it;
    }
    if (it == it_end) {
      break;
    }
    if (it->contains(date)) {
      flags(ix) = true;
    }
    ++ix;
  }

  return flags;
}

// ============================================================================

auto PeriodList::join_adjacent_periods(int64_t epsilon) const -> PeriodList {
  if (size() <= 1) {
    return *this;
  }

  auto result = PeriodList();
  result.reserve(size());
  result.push_back(front());

  for (auto it = begin() + 1; it != end(); ++it) {
    auto dt = it->begin - (it - 1)->last;
    if (dt <= epsilon) {
      result.back().last = it->last;
    } else {
      result.push_back(*it);
    }
  }
  result.shrink_to_fit();
  return result;
}

}  // namespace pyinterp
