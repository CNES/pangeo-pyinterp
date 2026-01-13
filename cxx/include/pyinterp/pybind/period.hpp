// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/nanobind.h>

#include "pyinterp/dateutils.hpp"
#include "pyinterp/period.hpp"

namespace pyinterp::pybind {

/// @brief A Period with associated datetime64 resolution.
class Period : public pyinterp::Period {
 public:
  /// @brief Create a null period with default resolution (datetime64[ns]).
  constexpr Period() = default;

  /// Create a Period from a base Period and a resolution.
  constexpr Period(const pyinterp::Period& period,
                   const dateutils::DType& resolution)
      : pyinterp::Period(period), resolution_(resolution) {}

  /// @brief Create a period with specified begin, end and resolution.
  /// @param[in] begin Start of the period (inclusive).
  /// @param[in] end End of the period.
  /// @param[in] resolution The numpy dtype resolution for this period.
  /// @param[in] within If true, end is inclusive; if false, end is exclusive.
  constexpr Period(int64_t begin, int64_t end, dateutils::DType resolution,
                   bool within = true) noexcept
      : pyinterp::Period(begin, end, within), resolution_(resolution) {}

  /// @brief Create a period with specified begin, end and resolution.
  /// @param[in] begin Start of the period (inclusive).
  /// @param[in] end End of the period.
  /// @param[in] within If true, end is inclusive; if false, end is exclusive.
  Period(const nanobind::object& begin, const nanobind::object& end,
         bool within);

  /// @brief Get the resolution of the period list.
  /// @return The resolution dtype.
  [[nodiscard]] constexpr auto resolution() const -> dateutils::DType {
    return resolution_;
  }

  /// @brief Convert this period to a different resolution.
  /// @param[in] target The target dtype resolution.
  /// @return A new Period converted to the target resolution.
  /// @throws std::overflow_error if conversion would overflow int64_t.
  [[nodiscard]] auto convert_to(const dateutils::DType& target) const
      -> Period {
    if (resolution_ == target) {
      return *this;
    }
    return Period{dateutils::convert(begin, resolution_, target),
                  dateutils::convert(last, resolution_, target), target};
  }

  /// @brief Equality comparison with resolution promotion.
  [[nodiscard]] auto operator==(const Period& rhs) const -> bool;

  /// @brief Three-way comparison with resolution promotion.
  [[nodiscard]] auto operator<=>(const Period& rhs) const
      -> std::strong_ordering;

  /// @brief Check if a point is contained within the period.
  /// @param[in] point A numpy.datetime64 scalar.
  /// @return true if the point is within the period, false otherwise.
  [[nodiscard]] auto contains(const nanobind::object& point) const -> bool;

  /// @brief Check if this period fully contains (or equals) the other period.
  /// @param[in] other The other period to check containment.
  /// @return true if this period contains the other, false otherwise.
  [[nodiscard]] auto contains(const Period& other) const -> bool;

  /// @brief Check if the entire period is after the given point
  /// (point < begin).
  /// @param[in] point A numpy.datetime64 scalar.
  /// @return true if the period is after the point, false otherwise.
  [[nodiscard]] auto is_after(const nanobind::object& point) const -> bool;

  /// @brief Check if the entire period is before the given point
  /// (last < point).
  /// @param[in] point A numpy.datetime64 scalar.
  /// @return true if the period is before the point, false otherwise.
  [[nodiscard]] auto is_before(const nanobind::object& point) const -> bool;

  /// @brief Check if the given date is within tolerance of the period.
  /// @param[in] date A numpy.datetime64 scalar.
  /// @param[in] tolerance The tolerance margin (inclusive) as a
  /// numpy.timedelta64 scalar.
  /// @return true if the date is within tolerance, false otherwise.
  [[nodiscard]] auto is_close(const nanobind::object& date,
                              const nanobind::object& tolerance) const -> bool;

  /// @brief Check if this period intersects with another period.
  /// @param[in] other The other period to check intersection with.
  /// @return true if the periods intersect, false otherwise.
  [[nodiscard]] auto intersects(const Period& other) const -> bool;

  /// @brief Check if this period is adjacent to another period.
  /// @param[in] other The other period to check adjacency with.
  /// @return true if the periods are adjacent, false otherwise.
  [[nodiscard]] auto is_adjacent(const Period& other) const -> bool;

  /// @brief Get the intersection of this period with another period.
  /// @param[in] other The other period to intersect with.
  /// @return The intersection period, or std::nullopt if disjoint.
  [[nodiscard]] auto intersection(const Period& other) const
      -> std::optional<Period>;

  /// @brief Merge this period with another period.
  /// @param[in] other The other period to merge with.
  /// @return The merged period, or std::nullopt if disjoint.
  [[nodiscard]] auto merge(const Period& other) const -> std::optional<Period>;

  /// @brief Extend the period to include the given point.
  /// @param[in] point A numpy.datetime64 scalar.
  /// @return A new Period extended to include the point.
  [[nodiscard]] auto extend(const nanobind::object& point) const -> Period;

  /// @brief Shift the period by a given timedelta.
  /// @param[in] offset A numpy.timedelta64 scalar.
  /// @return A new Period shifted by the offset.
  [[nodiscard]] auto shift(const nanobind::object& offset) const -> Period;

  /// @brief Convert the period to a string representation.
  /// @return String representation of the period.
  [[nodiscard]] explicit operator std::string() const;

  /// @brief Get the state of the object for pickling.
  /// @return A tuple representing the state of the object.
  [[nodiscard]] auto getstate() const
      -> std::tuple<int64_t, int64_t, std::string>;

  /// @brief Set the state of the object from pickling.
  /// @param[in] state A tuple representing the state of the object.
  static auto setstate(const std::tuple<int64_t, int64_t, std::string>& state)
      -> Period;

 private:
  /// The resolution of the period.
  dateutils::DType resolution_{};
};

class PeriodList : public pyinterp::PeriodList {
 public:
  /// @brief Create a null/default PeriodList.
  constexpr PeriodList() = default;

  /// @brief Create a PeriodList.
  /// @param[in] periods The Python list of periods.
  explicit PeriodList(const nanobind::list& periods);

  /// @brief Create a PeriodList from a matrix of begin/end values.
  /// @param[in] periods A 2D numpy array with shape (N, 2) containing begin/end
  /// values.
  /// @param[in] bool within If true, end is inclusive; if false, end is
  /// exclusive.
  explicit PeriodList(const nanobind::object& periods, bool within = true);

  /// @brief Append a period to the list.
  /// @param[in] period The period to append.
  auto append(const Period& period) -> void;

  /// @brief Set the period at the given index.
  /// @param[in] index The index to set.
  /// @param[in] period The period to set.
  auto setitem(size_t index, const Period& period) -> void;

  /// @brief Merge two lists of periods.
  /// @param[in] other The other PeriodList to merge with.
  /// @return A new PeriodList with merged periods.
  auto merge(const PeriodList& other) -> void;

  /// @brief Check if a date is within tolerance of any period in the list.
  /// @param[in] date A numpy.datetime64 scalar.
  /// @param[in] tolerance The tolerance margin as a numpy.timedelta64 scalar.
  /// @return true if the date is close to any period, false otherwise.
  [[nodiscard]] auto is_close(const nanobind::object& date,
                              const nanobind::object& tolerance) const -> bool;

  /// @brief Find the index of the period containing a date.
  /// @param[in] date A numpy.datetime64 scalar.
  /// @return Index of the containing period, or -1 if not found.
  [[nodiscard]] auto find_containing_index(const nanobind::object& date) const
      -> int64_t;

  /// @brief Find the period containing a date.
  /// @param[in] date A numpy.datetime64 scalar.
  /// @return The containing Period, or None if not found.
  [[nodiscard]] auto find_containing(const nanobind::object& date) const
      -> std::optional<Period>;

  /// @copydoc pyinterp::PeriodList::filter_contained
  [[nodiscard]] auto filter_contained(const Period& period) const noexcept
      -> PeriodList;

  /// @copydoc pyinterp::PeriodList::filter_min_duration
  [[nodiscard]] auto filter_min_duration(
      const nanobind::object& min_duration) const noexcept -> PeriodList;

  /// @copydoc pyinterp::PeriodList::cross_a_period
  [[nodiscard]] auto cross_a_period(const nanobind::object& dates) const
      -> Vector<bool>;

  /// @copydoc pyinterp::PeriodList::belong_to_a_period
  [[nodiscard]] auto belong_to_a_period(const nanobind::object& dates) const
      -> Vector<bool>;

  /// @copydoc pyinterp::PeriodList::join_adjacent_periods
  [[nodiscard]] inline auto join_adjacent_periods(
      const nanobind::object& epsilon) const -> PeriodList;

  /// @brief Get the state of the object for pickling.
  /// @return A tuple representing the state of the object.
  [[nodiscard]] auto getstate() const -> nanobind::tuple;

  /// @brief Set the state of the object from pickling.
  /// @param[in] state A tuple representing the state of the object.
  [[nodiscard]] static auto setstate(const nanobind::tuple& state)
      -> PeriodList;

  /// @brief Get the resolution of the period list.
  /// @return The resolution dtype.
  [[nodiscard]] constexpr auto resolution() const -> const dateutils::DType {
    return resolution_;
  }

  /// @brief Convert the period list to a string representation.
  /// @return String representation of the period list.
  [[nodiscard]] explicit operator std::string() const;

  /// @brief Access element by index (returns a Period with resolution).
  /// @note This returns a temporary Period object, not a reference to avoid
  /// undefined behavior from reinterpret_cast.
  [[nodiscard]] auto operator[](size_t index) const -> Period {
    return {pyinterp::PeriodList::operator[](index), resolution_};
  }

  /// @brief Get element at index (const, returns Period with resolution).
  [[nodiscard]] auto at(size_t index) const -> Period {
    if (index >= size()) {
      throw std::out_of_range("PeriodList index out of range");
    }
    return {pyinterp::PeriodList::operator[](index), resolution_};
  }

 private:
  /// The resolution of the periods in the list.
  dateutils::DType resolution_{};
  /// Magic number for serialization versioning.
  static constexpr uint32_t kMagicNumber = 0x504C4953;  // 'PLIS'

  /// @brief Construct a PeriodList from a base PeriodList and a resolution.
  /// @param[in,out] period_list The base PeriodList.
  /// @param[in] resolution The numpy dtype resolution for the periods.
  constexpr PeriodList(pyinterp::PeriodList&& period_list,
                       const dateutils::DType& resolution)
      : pyinterp::PeriodList(std::move(period_list)), resolution_(resolution) {}
};

/// @brief Initialize the Period classes in the given module.
/// @param[in,out] m The nanobind module to initialize.
auto init_period(nanobind::module_& m) -> void;

}  // namespace pyinterp::pybind
