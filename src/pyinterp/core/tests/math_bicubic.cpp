// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/detail/math/bicubic.hpp"
#include <gtest/gtest.h>

namespace math = pyinterp::detail::math;
namespace gsl = pyinterp::detail::gsl;

TEST(math_bicubic, xarray) {
  auto xr = math::XArray(3, 4);
  ASSERT_EQ(xr.nx(), 3);
  ASSERT_EQ(xr.ny(), 4);

  for (auto ix = 0; ix < 6; ++ix) {
    xr.x(ix) = ix * 2;
  }

  for (auto ix = 0; ix < 8; ++ix) {
    xr.y(ix) = ix * 2 + 1;
  }

  for (auto ix = 0; ix < 6; ++ix) {
    for (auto iy = 0; iy < 8; ++iy) {
      xr.z(ix, iy) = ix * iy;
    }
  }

  for (auto ix = 0; ix < 6; ++ix) {
    EXPECT_EQ(xr.x(ix), ix * 2);
  }

  for (auto ix = 0; ix < 8; ++ix) {
    EXPECT_EQ(xr.y(ix), ix * 2 + 1);
  }

  for (auto ix = 0; ix < 6; ++ix) {
    for (auto iy = 0; iy < 8; ++iy) {
      EXPECT_EQ(xr.z(ix, iy), ix * iy);
    }
  }

  EXPECT_EQ(xr.normalize_angle(-180), 180);
}

TEST(math_bicubic, bicubic) {
  auto xr = math::XArray(3, 3);
  ASSERT_EQ(xr.nx(), 3);
  ASSERT_EQ(xr.ny(), 3);

  for (auto ix = 0; ix < 6; ++ix) {
    xr.x(ix) = xr.y(ix) = ix * 0.1;
    for (auto iy = 0; iy < 6; ++iy) {
      xr.z(ix, iy) = std::sin(ix * 0.1);
    }
  }

  auto interpolator = math::Bicubic();
  auto acc = gsl::Accelerator();
  for (auto ix = 0; ix < 6; ++ix) {
    for (auto iy = 0; iy < 6; ++iy) {
      EXPECT_EQ(interpolator.interpolate(ix * 0.1, iy * 0.1, xr, acc),
                std::sin(ix * 0.1));
    }
  }
}
