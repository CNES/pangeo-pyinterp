#pragma once
#include <memory>
#include <stdexcept>

#include "pyinterp/detail/interpolation/bicubic.hpp"
#include "pyinterp/detail/interpolation/bilinear.hpp"

namespace pyinterp::detail::interpolation {

template <typename T>
static inline auto factory_2d(const std::string &kind)
    -> std::unique_ptr<Interpolator2D<T>> {
  if (kind == "bilinear") {
    return std::make_unique<Bilinear<T>>();
  }
  if (kind == "bicubic") {
    return std::make_unique<Bicubic<T>>();
  }
  throw std::invalid_argument("Invalid bicubic type: " + kind);
}

}  // namespace pyinterp::detail::interpolation
