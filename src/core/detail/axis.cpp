#include "pyinterp/detail/axis.hpp"

namespace pyinterp {
namespace detail {

void Axis::normalize_longitude(std::vector<double>& points) {
  auto monotonic = true;
  auto ascending = points.size() < 2 ? true : points[0] < points[1];

  for (size_t ix = 1; ix < points.size(); ++ix) {
    monotonic =
        ascending ? points[ix - 1] < points[ix] : points[ix - 1] > points[ix];

    if (!monotonic) {
      break;
    }
  }

  if (!monotonic) {
    auto cross = false;

    for (size_t ix = 1; ix < points.size(); ++ix) {
      if (!cross) {
        cross = ascending ? points[ix - 1] > points[ix]
                          : points[ix - 1] < points[ix];
      }

      if (cross) {
        points[ix] += ascending ? circle_ : -circle_;
      }
    }
  }
}

std::optional<double> is_evenly_spaced(const std::vector<double>& points,
                                       const double epsilon) {
  size_t n = points.size();

  // The axis is defined by a single value.
  if (n < 2) {
    return {};
  }

  double increment = (points.back() - points.front()) / (n - 1);

  for (size_t ix = 1; ix < n; ++ix) {
    if (!math::is_same(points[ix] - points[ix - 1], increment, epsilon)) {
      return {};
    }
  }
  return increment;
}

void Axis::compute_properties(const double epsilon) {
  // An axis can be represented by an empty set of values
  if (axis_->size() == 0) {
    throw std::invalid_argument("unable to create an empty axis.");
  }
  // Axis values must be sorted.
  if (!axis_->is_monotonic()) {
    throw std::invalid_argument("axis values are not ordered");
  }
  // If this axis represents an angle, determine if it represents the entire
  // trigonometric circle.
  if (is_angle()) {
    auto ptr = dynamic_cast<axis::container::Regular*>(axis_.get());
    if (ptr != nullptr) {
      is_circle_ = math::is_same<double>(std::fabs(ptr->step() * size()),
                                         circle_, epsilon);
    } else {
      auto increment = (axis_->back() - axis_->front()) / (axis_->size() - 1);
      is_circle_ = std::fabs((axis_->max_value() - axis_->min_value()) -
                             circle_) <= increment;
    }
  }
}

Axis::Axis(std::vector<double> values, const double epsilon,
           const bool is_circle, const bool is_radian)
    : circle_(Axis::set_circle(is_circle, is_radian)) {
  // Axis size control
  if (values.size() >
      static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument(
        "The size of the axis must not contain more than " +
        std::to_string(std::numeric_limits<int64_t>::max()) + "elements.");
  }

  if (is_angle()) {
    normalize_longitude(values);
  }

  // Determines whether the set of data provided can be represented as an
  // interval.
  auto increment = is_evenly_spaced(values, epsilon);
  if (increment) {
    axis_ = std::make_shared<axis::container::Regular>(axis::container::Regular(
        values.front(), values.back(), static_cast<double>(values.size())));
  } else {
    axis_ = std::make_shared<axis::container::Irregular>(
        axis::container::Irregular(std::move(values)));
  }
  compute_properties(epsilon);
}

std::optional<std::tuple<int64_t, int64_t>> Axis::find_indexes(
    double coordinate) const {
  coordinate = normalize_coordinate(coordinate);
  auto length = size();
  auto i0 = find_index(coordinate);

  /// If the value is outside the circle, then the value is between the last and
  /// first index.
  if (i0 == -1) {
    return is_circle_ ? std::make_tuple(static_cast<int64_t>(length - 1), 0LL)
                      : std::optional<std::tuple<int64_t, int64_t>>();
  }

  // Given the delta between the found coordinate and the given coordinate,
  // chose the other index that frames the coordinate
  auto delta = coordinate - (*this)(i0);
  auto i1 = i0;
  if (delta == 0) {
    // The requested coordinate is located on an element of the axis.
    i1 == length - 1 ? --i0 : ++i1;
  } else {
    if (delta < 0) {
      // The found point is located after the coordinate provided.
      is_ascending() ? --i0 : ++i0;
      if (is_circle_) {
        i0 = math::remainder(i0, length);
      }
    } else {
      // The found point is located before the coordinate provided.
      is_ascending() ? ++i1 : --i1;
      if (is_circle_) {
        i1 = math::remainder(i1, length);
      }
    }
  }

  if (i0 >= 0 && i0 < length && i1 >= 0 && i1 < length) {
    return std::make_tuple(i0, i1);
  }
  return std::optional<std::tuple<int64_t, int64_t>>{};
}

std::vector<int64_t> Axis::find_indexes(double coordinate, uint32_t size,
                                        Boundary boundary) const {
  if (size == 0) {
    throw std::invalid_argument("The size must not be zero.");
  }

  // Axis size
  auto len = this->size();

  // Searches the initial indexes and populate the result
  auto indexes = find_indexes(coordinate);
  if (!indexes) {
    return {};
  }
  auto result = std::vector<int64_t>(size << 1U);
  std::tie(result[size - 1], result[size]) = *indexes;

  // Offset in relation to the first indexes found
  uint32_t shift = 1;

  // Construction of window indexes based on the initial indexes found
  while (shift < size) {
    int64_t before = std::get<0>(*indexes) - shift;
    if (before < 0) {
      if (!is_circle_) {
        switch (boundary) {
          case kExpand:
            before = 0;
            break;
          case kWrap:
            before = math::remainder(len + before, len);
            break;
          case kSym:
            before = math::remainder(-before, len);
            break;
          default:
            return {};
        }
      } else {
        before = math::remainder(before, len);
      }
    }
    int64_t after = std::get<1>(*indexes) + shift;
    if (after >= len) {
      if (!is_circle_) {
        switch (boundary) {
          case kExpand:
            after = len - 1;
            break;
          case kWrap:
            after = math::remainder(after, len);
            break;
          case kSym:
            after = len - 2 - math::remainder(after - len, len);
            break;
          default:
            return {};
        }
      } else {
        after = math::remainder(after, len);
      }
    }
    result[size - shift - 1] = before;
    result[size + shift] = after;
    ++shift;
  }
  return result;
}

}  // namespace detail
}  // namespace pyinterp