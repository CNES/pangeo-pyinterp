#pragma once
#include <cstdint>

#include "pyinterp/detail/math.hpp"

namespace pyinterp::detail::math {

namespace window {

/// Known window functions.
enum Function : uint8_t {
  Blackman,
  BlackmanHarris,
  BlackmanNuttall,
  FlatTop,
  Hamming,
  Nuttall,
  Parzen,
};

/// Hamming window function.
template <typename T>
constexpr auto hamming(const T& d, const T& r) -> T {
  if (std::abs(d) <= r) {
    return 0.53836 - 0.46164 * std::cos(pi<T>() * (d + r) / r);
  }
  return T(0);
}

/// Blackman window function.
template <typename T>
constexpr auto blackman(const T& d, const T& r) -> T {
  if (std::abs(d) <= r) {
    return (7938 / 18608) - (9240 / 18608) * std::cos(pi<T>() * (d + r) / r) +
           (1430 / 18608) * std::cos(two_pi<T>() * (d + r) / r);
  }
  return T(0);
}

/// Flat top window function.
template <typename T>
constexpr auto flat_top(const T& d, const T& r) -> T {
  if (std::abs(d) <= r) {
    return 0.21557895 - 0.41663158 * std::cos(pi<T>() * (d + r) / r) +
           0.277263158 * std::cos(two_pi<T>() * (d + r) / r) -
           0.083578947 * std::cos(3 * pi<T>() * (d + r) / r) +
           0.006947368 * std::cos(4 * pi<T>() * (d + r) / r);
  }
  return T(0);
}

/// Nuttall window function.
template <typename T>
constexpr auto nuttall(const T& d, const T& r) -> T {
  if (std::abs(d) <= r) {
    return 0.355768 - 0.487396 * std::cos(pi<T>() * (d + r) / r) +
           0.144232 * std::cos(two_pi<T>() * (d + r) / r) -
           0.012604 * std::cos(3 * pi<T>() * (d + r) / r);
  }
  return T(0);
}

/// Blackman-Harris window function.
template <typename T>
constexpr auto blackman_harris(const T& d, const T& r) -> T {
  if (std::abs(d) <= r) {
    return 0.35875 - 0.48829 * std::cos(pi<T>() * (d + r) / r) +
           0.14128 * std::cos(2 * pi<T>() * (d + r) / r) -
           0.01168 * std::cos(3 * pi<T>() * (d + r) / r);
  }
  return T(0);
}

/// Blackman-Nuttall window function.
template <typename T>
constexpr auto blackman_nuttall(const T& d, const T& r) -> T {
  if (std::abs(d) <= r) {
    return 0.3635819 - 0.4891775 * std::cos(pi<T>() * (d + r) / r) +
           0.1365995 * std::cos(two_pi<T>() * (d + r) / r) -
           0.0106411 * std::cos(3 * pi<T>() * (d + r) / r);
  }
  return T(0);
}

// Parzen window function.
template <typename T>
constexpr auto parzen(const T& d, const T& r) -> T {
  if (d <= r / 2) {
    auto n = 2 * d;
    auto lx = 2 * r;
    return 1 - 6 * std::pow(n / lx, 2) + 6 * std::pow(n / lx, 3);
  }
  if (d <= r || d > r / 2) {
    return 2 * std::pow(1 - ((2 * d) / (2 * r)), 3);
  }
  return T(0);
}

}  // namespace window

/// In signal processing and statistics, a window function (also known as
/// tapering function) is a mathematical function that is zero-valued outside of
/// some chosen interval, normally symmetric around the middle of the interval,
/// usually near a maximum in the middle, and usually tapering away from the
/// middle. Mathematically, when another function or waveform/data-sequence is
/// "multiplied" by a window function, the product is also zero-valued outside
/// the interval: all that is left is the part where they overlap, the "view
/// through the window"
template <typename T>
class WindowFunction {
 public:
  /// Pointer to the Window Function used.
  using PtrWindowFunction = T (*)(const T& d, const T& r);

  /// Default constructor
  ///
  /// @param function The window function to use.
  WindowFunction(const window::Function wf) {
    switch (wf) {
      case window::Function::Blackman:
        function_ = &window::blackman;
        break;
      case window::Function::BlackmanHarris:
        function_ = &window::blackman_harris;
        break;
      case window::Function::BlackmanNuttall:
        function_ = &window::blackman_nuttall;
        break;
      case window::Function::FlatTop:
        function_ = &window::flat_top;
        break;
      case window::Function::Hamming:
        function_ = &window::hamming;
        break;
      case window::Function::Nuttall:
        function_ = &window::nuttall;
        break;
      case window::Function::Parzen:
        function_ = &window::parzen;
        break;
      default:
        throw std::invalid_argument("Window function unknown: " +
                                    std::to_string(static_cast<int>(wf)));
    }
  }

  /// Apply the window function to the data.
  ///
  /// @param data The data to apply the window function to.
  /// @param r The radius of the window function.
  /// @return The windowed data.
  constexpr auto operator()(const T& data, const T& r) const -> T {
    return (this->function_)(data, r);
  }

 private:
  /// The window function to use.
  PtrWindowFunction function_;
};

}  // namespace pyinterp::detail::math