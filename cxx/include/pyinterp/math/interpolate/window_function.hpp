// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <format>
#include <numbers>
#include <stdexcept>

#include "pyinterp/math.hpp"

namespace pyinterp::math::interpolate {
namespace window {

/// @brief Known window kernel for signal processing
enum class Kernel : uint8_t {
  kBlackman,        /// Blackman window
  kBlackmanHarris,  /// Blackman-Harris window
  kBoxcar,          /// Boxcar window
  kFlatTop,         /// Flat top window
  kGaussian,        /// Gaussian window
  kHamming,         /// Hamming window
  kLanczos,         /// Lanczos window
  kNuttall,         /// Nuttall window
  kParzen,          /// Parzen window
  kParzenSWOT,      /// Parzen SWOT window
};

/// @brief Hamming window function
/// @tparam T Floating-point type
/// @param[in] d Distance from center
/// @param[in] r Radius (half-width)
/// @return Window coefficient [0, 1]
template <std::floating_point T>
[[nodiscard]] constexpr auto hamming(const T d, const T r,
                                     [[maybe_unused]] const T arg) noexcept
    -> T {
  if (d <= r) [[likely]] {
    constexpr T a0 = T{0.53836};
    constexpr T a1 = T{0.46164};
    return a0 - a1 * std::cos(std::numbers::pi_v<T> * (d + r) / r);
  }
  return T{0};
}

/// @brief Blackman window function
/// @tparam T Floating-point type
/// @param[in] d Distance from center
/// @param[in] r Radius
/// @return Window coefficient [0, 1]
template <std::floating_point T>
[[nodiscard]] constexpr auto blackman(const T d, const T r,
                                      [[maybe_unused]] const T arg) noexcept
    -> T {
  if (d <= r) [[likely]] {
    const T ratio = (d + r) / r;
    constexpr T a0 = T{7938.0} / T{18608.0};
    constexpr T a1 = T{9240.0} / T{18608.0};
    constexpr T a2 = T{1430.0} / T{18608.0};
    return a0 - a1 * std::cos(std::numbers::pi_v<T> * ratio) +
           a2 * std::cos(T{2} * std::numbers::pi_v<T> * ratio);
  }
  return T{0};
}

/// @brief Flat top window function (used for accurate amplitude measurements)
/// @tparam T Floating-point type
/// @param[in] d Distance from center
/// @param[in] r Radius
/// @param[in] arg Additional argument (unused)
/// @return Window coefficient [0, 1]
template <std::floating_point T>
[[nodiscard]] constexpr auto flat_top(const T d, const T r,
                                      [[maybe_unused]] const T arg) noexcept
    -> T {
  if (d <= r) [[likely]] {
    const T ratio = (d + r) / r;
    constexpr T a0 = T{0.21557895};
    constexpr T a1 = T{0.41663158};
    constexpr T a2 = T{0.277263158};
    constexpr T a3 = T{0.083578947};
    constexpr T a4 = T{0.006947368};

    return a0 - a1 * std::cos(std::numbers::pi_v<T> * ratio) +
           a2 * std::cos(T{2} * std::numbers::pi_v<T> * ratio) -
           a3 * std::cos(T{3} * std::numbers::pi_v<T> * ratio) +
           a4 * std::cos(T{4} * std::numbers::pi_v<T> * ratio);
  }
  return T{0};
}

/// @brief Nuttall window function (continuous first derivative)
/// @tparam T Floating-point type
/// @param[in] d Distance from center
/// @param[in] r Radius
/// @param[in] arg Additional argument (unused)
/// @return Window coefficient [0, 1]
template <std::floating_point T>
[[nodiscard]] constexpr auto nuttall(const T d, const T r,
                                     [[maybe_unused]] const T arg) noexcept
    -> T {
  if (d <= r) [[likely]] {
    const T ratio = (d + r) / r;
    constexpr T a0 = T{0.3635819};
    constexpr T a1 = T{0.4891775};
    constexpr T a2 = T{0.1365995};

    return a0 - a1 * std::cos(std::numbers::pi_v<T> * ratio) +
           a2 * std::cos(T{2} * std::numbers::pi_v<T> * ratio);
  }
  return T{0};
}
/// @brief Blackman-Harris window function (improved sidelobe attenuation)
/// @tparam T Floating-point type
/// @param[in] d Distance from center
/// @param[in] r Radius
/// @param[in] arg Additional argument (unused)
/// @return Window coefficient [0, 1]
template <std::floating_point T>
[[nodiscard]] constexpr auto blackman_harris(
    const T d, const T r, [[maybe_unused]] const T arg) noexcept -> T {
  if (d <= r) [[likely]] {
    const T ratio = (d + r) / r;
    constexpr T a0 = T{0.35875};
    constexpr T a1 = T{0.48829};
    constexpr T a2 = T{0.14128};
    constexpr T a3 = T{0.01168};

    return a0 - a1 * std::cos(std::numbers::pi_v<T> * ratio) +
           a2 * std::cos(T{2} * std::numbers::pi_v<T> * ratio) -
           a3 * std::cos(T{3} * std::numbers::pi_v<T> * ratio);
  }
  return T{0};
}

/// @brief Lanczos window function (sinc-based)
/// @tparam T Floating-point type
/// @param[in] d Distance from center
/// @param[in] r Radius
/// @param[in] nlobes Number of lobes (typically 2-3)
template <std::floating_point T>
[[nodiscard]] constexpr auto lanczos(const T d, const T r,
                                     const T nlobes) noexcept -> T {
  if (d <= nlobes * r) [[likely]] {
    return math::sinc(d / r) * math::sinc(d / (r * nlobes));
  }
  return T{0};
}

/// @brief Gaussian window function
/// @tparam T Floating-point type
/// @param[in] d Distance from center
/// @param[in] r Radius
/// @param[in] sigma Standard deviation parameter
template <std::floating_point T>
[[nodiscard]] constexpr auto gaussian(const T d, const T r,
                                      const T sigma) noexcept -> T {
  if (d <= r) [[likely]] {
    return std::exp(-T{0.5} * math::sqr(d / sigma));
  }
  return T{0};
}

/// @brief Parzen window function (piecewise cubic)
/// @tparam T Floating-point type
/// @param[in] d Distance from center
/// @param[in] r Radius
/// @param[in] sampling Sampling parameter
template <std::floating_point T>
[[nodiscard]] constexpr auto parzen(const T d, const T r,
                                    const T sampling) noexcept -> T {
  const T ratio = d / r;
  const T l = T{2} * r + sampling;
  const T l_4 = l / T{4};
  const T l_2 = l / T{2};

  if (d <= l_4) {
    return T{1} - T{6} * ratio * ratio * (T{1} - ratio);
  }
  if (d > l_4 && d <= l_2) {
    const T one_minus_ratio = T{1} - ratio;
    return T{2} * one_minus_ratio * one_minus_ratio * one_minus_ratio;
  }
  return T{0};
}

/// @brief Parzen-like window function used for SWOT satellite products
/// @tparam T Floating-point type
/// @param[in] d Distance from center
/// @param[in] r Radius
/// @param[in] arg Additional argument (unused)
/// @return Window coefficient [0, 1]
template <std::floating_point T>
[[nodiscard]] constexpr auto parzen_swot(const T d, const T r,
                                         [[maybe_unused]] const T arg) noexcept
    -> T {
  const T l = T{2} * r;
  const T ratio = (T{2} * d) / l;
  const T l_4 = l / T{4};
  const T l_2 = l / T{2};

  if (d <= l_4) {
    const T ratio2 = ratio * ratio;
    return T{1} - T{6} * ratio2 + T{6} * ratio2 * ratio;
  }
  if (d > l_4 && d <= l_2) {
    const T one_minus_ratio = T{1} - ratio;
    return T{2} * one_minus_ratio * one_minus_ratio * one_minus_ratio;
  }
  return T{0};
}

/// Boxcar (rectangular) window function
template <std::floating_point T>
[[nodiscard]] constexpr auto boxcar(const T d, const T r,
                                    [[maybe_unused]] const T arg) noexcept
    -> T {
  return d <= r ? T{1} : T{0};
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
template <std::floating_point T>
class InterpolationWindow {
 public:
  /// Pointer to window function
  using WindowFunctionPtr = T (*)(T, T, T) noexcept;

  /// @brief Constructor
  /// @param[in] wf The window function type to use
  /// @param[in] arg Optional argument for the window function. Defaults to 0.
  /// Its meaning depends on the window function selected.
  explicit InterpolationWindow(const window::Kernel wf, const T arg = T{0})
      : arg_(arg) {
    switch (wf) {
      case window::Kernel::kBlackman:
        function_ = &window::blackman<T>;
        break;
      case window::Kernel::kBlackmanHarris:
        function_ = &window::blackman_harris<T>;
        break;
      case window::Kernel::kBoxcar:
        function_ = &window::boxcar<T>;
        break;
      case window::Kernel::kFlatTop:
        function_ = &window::flat_top<T>;
        break;
      case window::Kernel::kLanczos:
        function_ = &window::lanczos<T>;
        break;
      case window::Kernel::kGaussian:
        function_ = &window::gaussian<T>;
        break;
      case window::Kernel::kHamming:
        function_ = &window::hamming<T>;
        break;
      case window::Kernel::kNuttall:
        function_ = &window::nuttall<T>;
        break;
      case window::Kernel::kParzen:
        function_ = &window::parzen<T>;
        break;
      case window::Kernel::kParzenSWOT:
        function_ = &window::parzen_swot<T>;
        break;
      [[unlikely]] default:
        throw std::invalid_argument(
            std::format("Window function unknown: {}", static_cast<int>(wf)));
    }

    if (arg_ == 0 &&
        (wf == window::Kernel::kGaussian || wf == window::Kernel::kLanczos)) {
      throw std::invalid_argument(
          "Window function argument must be non-zero for Gaussian and "
          "Lanczos windows.");
    }
  }

  /// @brief Apply the window function to the data
  /// @tparam T Floating-point type
  /// @param[in] data Distance from window center
  /// @param[in] r Radius (half-width) of the window
  /// @return Window coefficient [0, 1]
  [[nodiscard]] constexpr auto operator()(const T data,
                                          const T r) const noexcept -> T {
    return function_(data, r, arg_);
  }

 private:
  /// Pointer to the selected window function
  WindowFunctionPtr function_;
  /// Additional argument for the window function
  T arg_;
};

}  // namespace pyinterp::math::interpolate
