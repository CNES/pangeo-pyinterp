// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/detail/interpolation/akima.hpp"

namespace pyinterp::detail::interpolation {

/// Akima periodic interpolation
template <typename T>
class AkimaPeriodic : public Akima<T> {
 public:
  using Akima<T>::Akima;

 private:
  /// Compute the boundary conditions.
  auto boundary_condition(T* m, const size_t size) -> void override {
    m[-2] = m[size - 3];
    m[-1] = m[size - 2];
    m[size - 1] = m[0];
    m[size] = m[1];
  }
};

}  // namespace pyinterp::detail::interpolation
