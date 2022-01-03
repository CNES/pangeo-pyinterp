// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/math/bicubic.hpp"

namespace math = pyinterp::detail::math;
namespace gsl = pyinterp::detail::gsl;

TEST(math_bicubic, bicubic) {
  auto xr = math::Frame2D(2, 2);

  auto xarr = std::vector<double>{0.0, 1.0, 2.0, 3.0};
  auto yarr = std::vector<double>{0.0, 1.0, 2.0, 3.0};
  auto zarr = std::vector<double>{1.0, 1.1, 1.2, 1.3, 1.1, 1.2, 1.3, 1.4,
                                  1.2, 1.3, 1.4, 1.5, 1.3, 1.4, 1.5, 1.6};
  auto xval = std::vector<double>{1.0, 1.5, 2.0};
  auto yval = std::vector<double>{1.0, 1.5, 2.0};
  auto zval = std::vector<double>{1.2, 1.3, 1.4};

  for (auto ix = 0; ix < 4; ++ix) {
    xr.x(ix) = xarr[ix];
    for (auto jx = 0; jx < 4; ++jx) {
      xr.q(ix, jx) = zarr[jx * 4 + ix];
    }
  }

  for (auto jx = 0; jx < 4; ++jx) {
    xr.y(jx) = yarr[jx];
  }

  auto interpolator = math::Bicubic(xr, "bicubic");
  for (auto ix = 0; ix < 3; ++ix) {
    EXPECT_NEAR(interpolator.interpolate(xval[ix], yval[ix], xr), zval[ix],
                1e-10);
  }
}
