#pragma once
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <cstdint>
#include <optional>

namespace pyinterp {

/// Represents a period of time.
struct Period {
  /// Create a period from begin to last eg: [begin,end)
  constexpr Period(const int64_t begin, const int64_t end,
                   const bool within = true)
      : begin(begin), last(end - (within ? 0 : 1)) {}

  /// Create a null period.
  constexpr Period() = default;

  /// Return one past the last element.
  constexpr auto end() const -> int64_t { return last + 1; }

  /// Return the length of the period.
  constexpr auto length() const -> int64_t {
    if (last < begin) {  // invalid period
      return last + 1 - begin;
    } else {
      return end() - begin;  // normal case
    }
  }

  constexpr auto duration() const -> int64_t { return end() - begin; }

  /// True if period is ill formed (length is zero or less)
  constexpr auto is_null() const -> bool { return end() <= begin; }

  /// Equality operator.
  constexpr auto operator==(const Period &rhs) const -> bool {
    return ((begin == rhs.begin) && (last == rhs.last));
  }

  /// Strict as defined by rhs.last <= lhs.last.
  constexpr auto operator<(const Period &rhs) const -> bool {
    return (last < rhs.begin);
  }

  /// True if the point is inside the period, zero length periods contain no
  /// points
  constexpr auto contains(const int64_t point) const -> bool {
    return ((point >= begin) && (point <= last));
  }

  /// True if this period fully contains (or equals) the other period
  constexpr auto contains(const Period &other) const -> bool {
    return ((begin <= other.begin) && (last >= other.last));
  }

  /// True if periods are next to each other without a gap.
  ///
  /// In the example below, p1 and p2 are adjacent, but p3 is not adjacent
  /// with either of p1 or p2.
  /// @code
  ///    [-p1-)
  ///         [-p2-)
  ///           [-p3-)
  /// @endcode
  constexpr auto is_adjacent(const Period &other) const -> bool {
    return (other.begin == end() || begin == other.end());
  }
  /// True if all of the period is prior or point < start
  /// In the example below only point 1 would evaluate to true.
  /// @code
  ///      [---------])
  ///  ^   ^    ^     ^   ^
  ///  1   2    3     4   5
  ///
  /// @endcode
  constexpr auto is_after(const int64_t point) const -> bool {
    if (is_null()) {
      return false;  // null period isn't after
    }

    return point < begin;
  }

  /// True if all of the period is prior to the passed point or end <= t
  /// In the example below points 4 and 5 return true.
  ///@code
  ///     [---------])
  /// ^   ^    ^     ^   ^
  /// 1   2    3     4   5
  ///
  ///@endcode
  constexpr auto is_before(const int64_t point) const -> bool {
    if (is_null()) {
      return false;  // null period isn't before anything
    }

    return last < point;
  }

  /// True if the periods overlap in any way
  ///  In the example below p1 intersects with p2, p4, and p6.
  /// @code
  ///        [---p1---)
  ///              [---p2---)
  ///                 [---p3---)
  ///   [---p4---)
  ///  [-p5-)
  ///          [-p6-)
  /// @endcode
  constexpr auto intersects(const Period &other) const -> bool {
    return (contains(other.begin) || other.contains(begin) ||
            ((other.begin < begin) && (other.last >= begin)));
  }

  /// Return the intersection of two periods.
  ///
  /// If the periods do not intersect, the result is a null period.
  /// @param other is the other period to intersect with.
  /// @returns the intersection of the two periods.
  inline auto intersection(const Period &other) const -> Period {
    if (begin > other.begin) {
      if (last <= other.last) {
        return *this;
      }
      return Period(begin, other.last);
    }

    if (last <= other.last) {
      return Period(other.begin, last);
    }
    return other;
  }

  /// Returns the union of intersecting periods -- or null period
  ///
  /// @param other is the other period to union with.
  /// @returns the union of the two periods.
  auto merge(const Period &other) const -> Period {
    if (intersects(other)) {
      if (begin < other.begin) {
        return Period(begin, last > other.last ? last : other.last);
      }
      return Period(other.begin, last > other.last ? last : other.last);
    }
    // no intersect return null
    return Period(begin, begin, false);
  }

  /// Return true if the given date is close to the period.
  ///
  /// @param date is the date to check.
  /// @param tolerance The maximum distance from the period to consider the date
  /// close.
  /// @returns true if the date is close to the period.
  constexpr auto is_close(const int64_t date, const int64_t tolerance) const
      -> bool {
    return (date >= begin - tolerance) && (date <= last + tolerance);
  }

  /// The start time of the period.
  int64_t begin{0};
  /// The last time in the period.
  int64_t last{0};
};

/// The list of periods.
class PeriodList {
 public:
  /// Default constructor.
  PeriodList(Eigen::Matrix<Period, -1, 1> periods)
      : periods_(std::move(periods)) {}

  /// Return the number of periods in the list.
  constexpr auto size() const -> int64_t { return periods_.size(); }

  /// Return true if the list is empty.
  constexpr auto empty() const -> bool { return size() == 0; }

  /// Check that the periods are sorted in ascending order and disjoint from
  /// each other.
  auto are_periods_sorted_and_disjointed() const -> bool {
    auto size = periods_.size();
    if (size > 1) {
      auto begin = periods_.array().begin();
      auto previous = begin;
      for (auto next = begin + 1; next < begin + size; ++next) {
        if (previous->intersects(*next) || next->last < previous->begin) {
          return false;
        }
        previous = next;
      }
    }
    return true;
  }

  /// Sort the periods in ascending order.
  inline auto sort() -> void {
    std::sort(periods_.array().begin(), periods_.array().end(),
              [](auto &lhs, auto &rhs) { return lhs.begin < rhs.begin; });
  }

  //// Merge two list of periods.
  auto merge(const PeriodList &other) -> void {
    auto periods =
        Eigen::Matrix<Period, -1, 1>(periods_.size() + other.periods_.size());
    auto ix = int64_t(0);
    auto jx = int64_t(0);
    auto kx = int64_t(0);

    auto insert_or_merge = [&kx, &periods](const Period &period) -> void {
      if (kx == 0) {
        periods(kx++) = period;
      } else {
        auto overlap = periods(kx - 1).merge(period);
        if (overlap.is_null()) {
          periods(kx++) = period;
        } else {
          periods(kx - 1) = overlap;
        }
      }
    };

    while (ix < periods_.size() && jx < other.periods_.size()) {
      insert_or_merge(other.periods_[jx].is_after(periods_[ix].begin)
                          ? periods_[ix++]
                          : other.periods_[jx++]);
    }

    while (ix < periods_.size()) {
      insert_or_merge(periods_[ix++]);
    }

    while (jx < other.periods_.size()) {
      insert_or_merge(other.periods_[jx++]);
    }

    periods.conservativeResize(kx);
    periods_ = std::move(periods);
  }

  /// Inspects the dates provided to determine if they belong to one of the
  /// given periods. Returns a boolean vector of same size as the input vectors.
  /// True if the date belongs to the period.
  ///
  /// The vectors provided should be sorted in chronological order.
  auto cross_a_period(
      const Eigen::Ref<const Eigen::Matrix<int64_t, -1, 1>> &dates) const
      -> Eigen::Matrix<bool, -1, 1> {
    // The index of the closest period of the current date processed.
    auto first_index = int64_t(0);

    // Flag equal to 1 if the date belongs to a period, 0 otherwise.
    auto flags = Eigen::Matrix<bool, -1, 1>(dates.size());
    flags.setConstant(false);

    // Index of the traversed date.
    auto ix = int64_t(0);

    // The last date processed.
    const auto last_date = dates[dates.size() - 1];

    // Searches for the period containing or after the provided date.
    auto lookup =
        [this](const int64_t first_index,
               const int64_t date) -> std::tuple<int64_t, const Period *> {
      for (auto index = first_index; index < periods_.size(); ++index) {
        const auto *period = &periods_[index];
        if (period->contains(date) || period->is_after(date)) {
          return {index, period};
        }
      }
      return {-1, nullptr};
    };

    // The index of the first period that is located after the last date
    // provided.
    auto [last_index, period] = lookup(0, last_date);
    if (last_index != -1 && period->contains(last_date)) {
      // If the last date processed belongs to a period, no dates are valid.
      flags.setConstant(true);
      return flags;
    }

    while (ix < dates.size()) {
      const auto date = dates(ix);

      std::tie(first_index, period) = lookup(first_index, date);
      if (first_index == -1 || first_index == last_index) {
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
      ++ix;
    }
    return flags;
  }

  /// Masks the dates provided to determine if they belong to one of the given
  /// periods. Returns a boolean vector of same size as the input vectors. True
  /// if the date belongs to the period.
  ///
  /// The vectors provided should be sorted in chronological order.
  auto belong_to_a_period(
      const Eigen::Ref<const Eigen::Matrix<int64_t, -1, 1>> &dates) const
      -> Eigen::Matrix<bool, -1, 1> {
    // Flag equal to 1 if the date belongs to a period, 0 otherwise.
    auto flags = Eigen::Matrix<bool, -1, 1>(dates.size());

    {
      auto gil = pybind11::gil_scoped_release();
      flags.setConstant(false);

      // Index of the traversed date.
      auto ix = int64_t(0);
      auto it = periods_.array().begin();
      auto end = periods_.array().end();

      while (ix < dates.size()) {
        const auto date = dates(ix);
        while (!it->contains(date) && !it->is_after(date) && it != end) {
          ++it;
        }
        if (it == end) {
          break;
        }
        if (it->contains(date)) {
          flags(ix) = true;
        }
        ++ix;
      }
    }
    return flags;
  }

  /// Get the periods.
  auto periods() const -> const Eigen::Matrix<Period, -1, 1> & {
    return periods_;
  }

  /// Determines whether the date, given in parameter, is close to a period in
  /// the list.
  ///
  /// @arg date The date to check.
  /// @arg epsilon The maximum distance from the period to consider the date
  /// close.
  /// @return True if the date is close to a period in the list.
  auto is_close(const int64_t date, const int64_t epsilon) const -> bool {
    for (const auto &period : periods_) {
      if (period.is_close(date, epsilon)) {
        return true;
      }
    }
    return false;
  }

  /// Join the periods together if they are adjacent.
  ///
  /// @arg epsilon The maximum gap between periods to join.
  /// @return A new PeriodList with the joined periods.
  auto join_adjacent_periods(const int64_t epsilon) const -> PeriodList {
    auto result = PeriodList(periods_);
    if (result.size() <= 1) {
      return result;
    }

    auto ix = int64_t(1);
    for (auto jx = int64_t(1); jx < result.periods_.size(); ++jx) {
      auto dt = periods_[jx].begin - periods_[jx - 1].last;
      if (dt <= epsilon) {
        result.periods_(ix - 1).last = periods_[jx].last;
      } else {
        result.periods_(ix) = periods_[jx];
        ++ix;
      }
    }
    result.periods_.conservativeResize(ix);
    return result;
  }

  /// Returns the list of periods that are within the given period.
  ///
  /// @arg period The period to search for.
  /// @return A new PeriodList with the periods that are within the given
  /// period.
  auto within(const Period &period) const -> PeriodList {
    auto result = PeriodList(periods_);
    auto ix = int64_t(0);
    for (auto jx = int64_t(0); jx < result.periods_.size(); ++jx) {
      if (period.intersects(result.periods_(jx))) {
        result.periods_(ix) = result.periods_(jx);
        ++ix;
      }
    }
    result.periods_.conservativeResize(ix);
    return result;
  }

  /// Returns the intersection of the list of periods with the given period.
  ///
  /// @arg other The period list to intersect with.
  /// @return A new PeriodList that intersects with the given period list.
  auto intersection(const Period &period) const -> PeriodList {
    auto result = PeriodList(periods_);
    auto ix = int64_t(0);
    for (auto jx = int64_t(0); jx < result.periods_.size(); ++jx) {
      auto intersection = result.periods_(jx).intersection(period);
      if (!intersection.is_null()) {
        result.periods_(ix) = intersection;
        ++ix;
      }
    }
    result.periods_.conservativeResize(ix);
    return result;
  }

  /// Returns the list of periods  whose minimum period is greater than or
  /// equal to the given period.
  ///
  /// @arg period The period to search for.
  /// @return A new PeriodList with the periods whose minimum period is greater
  /// than or equal to the given period.
  auto filter(int64_t min_duration) const -> PeriodList {
    auto result = PeriodList(periods_);
    auto ix = int64_t(0);
    for (auto jx = int64_t(0); jx < result.periods_.size(); ++jx) {
      if (periods_(jx).duration() >= min_duration) {
        result.periods_(ix) = result.periods_(jx);
        ++ix;
      }
    }
    result.periods_.conservativeResize(ix);
    return result;
  }

 private:
  Eigen::Matrix<Period, -1, 1> periods_;
};

}  // namespace pyinterp
