// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

namespace pyinterp::detail::math {

#ifdef HAVE_IEC559

// IEEE-754 Floating Point parts
struct FloatingPointParts {
  uint64_t mantissa : 52;
  uint64_t exponent : 11;
  uint64_t sign : 1;
};

// IEEE-754 Floating Point
union FloatingPoint {
  double value;
  FloatingPointParts parts;
};

// Fast calculation of 2^n
constexpr auto power2(const int32_t exponent) -> double {
  FloatingPoint floating_point{2};
  floating_point.parts.exponent += static_cast<uint64_t>(exponent) - 1;
  return floating_point.value;
}

// Fast calculation of 10^n
constexpr auto power10(int32_t exponent) -> double {
  auto result = 1.0;
  auto base = 10.0;
  bool inv = exponent < 0;
  if (inv) {
    exponent = -exponent;
  }
  while (exponent) {
    if (exponent & 1) {
      result *= base;
    }
    exponent >>= 1U;
    base *= base;
  }
  return inv ? 1.0 / result : result;
}
#else

// Calculate of 2^n
inline auto power2(const int32_t exponent) -> double {
  return std::pow(2.0, static_cast<double>(exponent));
}

// Calculate of 10^n
inline auto power10(int32_t exponent) -> double {
  return std::pow(10.0, static_cast<double>(exponent));
}

#endif

/// π
template <typename T>
constexpr auto pi() noexcept -> T {
  return std::atan2(T(0), T(-1));
}

/// π/2
template <typename T>
constexpr auto pi_2() noexcept -> T {
  return 0.5 * pi<T>();
}

/// 2π
template <typename T>
constexpr auto two_pi() noexcept -> T {
  return T(2) * pi<T>();
}

/// Convert angle x from radians to degrees.
template <typename T>
constexpr auto radians(const T &x) noexcept -> T {
  return x * pi<T>() / T(180);
}

/// Convert angle x from degrees to radians.
template <typename T>
constexpr auto degrees(const T &x) noexcept -> T {
  return x * T(180) / pi<T>();
}

/// Computes the remainder of the operation x/y
///
/// @return a result with the same sign as its second operand
template <typename T, typename std::enable_if<std::is_integral<T>::value,
                                              T>::type * = nullptr>
constexpr auto remainder(const T &x, const T &y) noexcept -> T {
  auto result = x % y;
  return result != 0 && (result ^ y) < 0 ? result + y : result;
}

/// Computes the remainder of the operation x/y
///
/// @return a result with the same sign as its second operand
template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              T>::type * = nullptr>
constexpr auto remainder(const T &x, const T &y) noexcept -> T {
  auto result = std::remainder(x, y);
  if (result < T(0)) {
    result += y;
  }
  return result;
}

/// Normalize an angle.
///
/// @param x The angle in degrees.
/// @param min Minimum circle value
/// @param circle Circle value
/// @return the angle reduced to the range [min, circle + min[
template <typename T>
constexpr auto normalize_angle(const T &x, const T &min,
                               const T &circle) noexcept -> T {
  return remainder(x - min, circle) + min;
}

/// Evaluate the sine function with the argument in degrees
///
/// In order to minimize round-off errors, this function exactly reduces the
/// argument to the range [-45, 45] before converting it to radians.
///
/// @param x x in degrees.
/// @return sin(x).
template <typename T>
constexpr auto sind(const T &x) noexcept -> T {
  int quotient{};
  T result = radians(std::remquo(x, T(90), &quotient));
  // now |result| <= π/4
  auto quadrant = static_cast<unsigned int>(quotient);
  result = (quadrant & 1U) ? std::cos(result) : std::sin(result);
  if (quadrant & 2U) {
    result = -result;
  }
  return result;
}

/// Evaluate the cosine function with the argument in degrees
///
/// @param x in degrees.
/// @return cos(x).
template <typename T>
constexpr auto cosd(const T &x) noexcept -> T {
  int quotient{};
  T result = radians(std::remquo(x, T(90), &quotient));
  // now |result| <= π/4
  auto quadrant = static_cast<unsigned int>(quotient + 1);
  result = (quadrant & 1U) ? std::cos(result) : std::sin(result);
  if (quadrant & 2U) {
    result = -result;
  }
  return result;
}

/// Evaluate the sine and cosine function with the argument in degrees
///
/// @param x in degrees.
/// @return a tuple that contains sin(x) and cos(x)
template <typename T>
constexpr auto sincosd(const T &x) noexcept -> std::tuple<T, T> {
  int quotient{};
  T angle = radians(std::remquo(x, T(90), &quotient));
  // now |angle| <= π/4
  switch (static_cast<unsigned int>(quotient) & 3U) {
    case 0U:
      return std::make_tuple(std::sin(angle), std::cos(angle));
    case 1U:
      return std::make_tuple(std::cos(angle), -std::sin(angle));
    case 2U:
      return std::make_tuple(-std::sin(angle), -std::cos(angle));
    // case 3U
    default:
      return std::make_tuple(-std::cos(angle), std::sin(angle));
  }
}

/// Evaluate the tangent function with the argument in degrees
///
/// @param x in degrees.
/// @return tan(x).
template <typename T>
constexpr auto tand(const T &x) noexcept -> T {
  auto [sinx, cosx] = sincosd(x);
  return cosx != 0 ? sinx / cosx : (sinx < 0 ? -HUGE_VAL : HUGE_VAL);
}

/// Evaluate the atan2 function with the result in degrees
///
/// @param y
/// @param x
/// @return atan2(y, x) in degrees.
template <typename T>
constexpr auto atan2d(T y, T x) noexcept -> T {
  // In order to minimize round-off errors, this function rearranges the
  // arguments so that result of atan2 is in the range [-π/4, π/4] before
  // converting it to degrees and mapping the result to the correct
  // quadrant.
  int quadrant = 0;
  if (std::abs(y) > std::abs(x)) {
    std::swap(x, y);
    quadrant = 2;
  }
  if (x < 0) {
    x = -x;
    ++quadrant;
  }
  // here x >= 0 and x >= abs(y), so angle is in [-π/4, π/4]
  T angle = degrees(std::atan2(y, x));
  switch (quadrant) {
    case 1:
      angle = (y >= 0 ? 180 : -180) - angle;
      break;
    case 2:
      angle = 90 - angle;
      break;
    case 3:
      angle = -90 + angle;
      break;
    default:
      break;
  }
  return angle;
}

/// Evaluate the atan function with the result in degrees
///
/// @param x
/// @return atan(x) in degrees.
template <typename T>
constexpr auto atand(const T &x) noexcept -> T {
  return atan2d(x, T(1));
}

/// Square a number.
///
/// @return \f$x^2\f$
template <typename T>
constexpr auto sqr(const T &x) noexcept -> T {
  return x * x;
}

/// True if a is almost zero to epsilon
template <typename T>
constexpr auto is_almost_zero(const T &a, const T &epsilon) noexcept -> bool {
  return std::fabs(a) < epsilon;
}

/// Return the normalized sinc function
template <typename T>
constexpr auto sinc(const T &x) noexcept -> T {
  return x == 0 ? T(1) : std::sin(pi<T>() * x) / (pi<T>() * x);
}

/// True if a and b are two values identical to an epsilon.
template <typename T, typename std::enable_if<std::is_integral<T>::value,
                                              T>::type * = nullptr>
constexpr auto is_same(const T &a, const T &b, const T &epsilon) noexcept
    -> bool {
  return std::abs(a - b) <= epsilon;
}

/// True if a and b are two values identical to an epsilon.
template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              T>::type * = nullptr>
constexpr auto is_same(const T &a, const T &b, const T &epsilon) noexcept
    -> bool {
  auto diff = std::fabs(a - b);
  if (diff <= epsilon) {
    return true;
  }
  if (diff < std::fmax(std::fabs(a), std::fabs(b)) * epsilon) {
    return true;
  }
  return false;
}

/// Compares two real values for a given accuracy expressed as a number of real
/// values that can be represented between these two values.
template <typename T>
constexpr auto is_within(const T &a, const T &b, size_t interval_size) -> bool {
  const T lower =
      (a - std::nextafter(a, std::numeric_limits<T>::lowest())) * interval_size;
  const T upper =
      (std::nextafter(a, std::numeric_limits<T>::max()) - a) * interval_size;

  return (a - lower) <= b && b <= (a + upper);
}

/// Represents a filling value
template <typename T, class Enable = void>
struct Fill;

/// Represents a filling value for floating point number
template <class T>
struct Fill<T, std::enable_if_t<std::is_floating_point<T>::value>> {
  static constexpr auto value() noexcept -> T {
    return std::numeric_limits<T>::quiet_NaN();
  }
  static constexpr auto is(const T &x) noexcept -> T { return std::isnan(x); }
  static constexpr auto is_not(const T &x) noexcept -> T {
    return !std::isnan(x);
  }
};

/// Represents a filling value for integer number
template <class T>
struct Fill<T, std::enable_if_t<std::is_integral<T>::value>> {
  static constexpr auto value() noexcept -> T {
    return std::numeric_limits<T>::max();
  }
  static constexpr auto is(const T &x) noexcept -> T {
    return Fill::value() == x;
  }
  static constexpr auto is_not(const T &x) noexcept -> T {
    return Fill::value() != x;
  }
};

}  // namespace pyinterp::detail::math
