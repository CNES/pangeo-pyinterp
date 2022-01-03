// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/math/spline2d.hpp"

namespace math = pyinterp::detail::math;
namespace gsl = pyinterp::detail::gsl;

TEST(math_spline2d, frame_2d) {
  auto xr = math::Frame2D(3, 4);
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
      xr.q(ix, iy) = ix * iy;
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
      EXPECT_EQ(xr.q(ix, iy), ix * iy);
    }
  }

  EXPECT_EQ(xr.normalize_angle(-180), 180);
}

TEST(math_spline2d, xarray_3d) {
  auto xr = math::Frame3D<int64_t>(3, 4, 1);
  ASSERT_EQ(xr.nx(), 3);
  ASSERT_EQ(xr.ny(), 4);
  ASSERT_EQ(xr.nz(), 1);

  for (auto ix = 0; ix < 6; ++ix) {
    xr.x(ix) = ix * 2;
  }

  for (auto ix = 0; ix < 8; ++ix) {
    xr.y(ix) = ix * 2 + 1;
  }

  for (auto ix = 0; ix < 6; ++ix) {
    for (auto iy = 0; iy < 8; ++iy) {
      for (auto iz = 0; iz < 2; ++iz) {
        xr.q(ix, iy, iz) = ix * iy * (iz + 1);
      }
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
      for (auto iz = 0; iz < 2; ++iz) {
        EXPECT_EQ(xr.q(ix, iy, iz), ix * iy * (iz + 1));
      }
    }
  }

  auto xr0 = xr.frame_2d(0);
  EXPECT_EQ(xr0.x(), xr.x());
  EXPECT_EQ(xr0.y(), xr.y());

  for (auto ix = 0; ix < 6; ++ix) {
    for (auto iy = 0; iy < 8; ++iy) {
      EXPECT_EQ(xr0.q(ix, iy), ix * iy);
    }
  }

  auto xr1 = xr.frame_2d(1);
  for (auto ix = 0; ix < 6; ++ix) {
    for (auto iy = 0; iy < 8; ++iy) {
      EXPECT_EQ(xr1.q(ix, iy), ix * iy * 2);
    }
  }
}

TEST(math_spline2d, xarray_4d) {
  auto xr = math::Frame4D<int64_t>(3, 4, 1, 1);
  ASSERT_EQ(xr.nx(), 3);
  ASSERT_EQ(xr.ny(), 4);
  ASSERT_EQ(xr.nz(), 1);
  ASSERT_EQ(xr.nu(), 1);

  for (auto ix = 0; ix < 6; ++ix) {
    xr.x(ix) = ix * 2;
  }

  for (auto ix = 0; ix < 8; ++ix) {
    xr.y(ix) = ix * 2 + 1;
  }

  for (auto ix = 0; ix < 6; ++ix) {
    for (auto iy = 0; iy < 8; ++iy) {
      for (auto iz = 0; iz < 2; ++iz) {
        for (auto iu = 0; iu < 2; ++iu) {
          xr.q(ix, iy, iz, iu) = ix * iy * (iz + 1) * (iu + 1);
        }
      }
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
      for (auto iz = 0; iz < 2; ++iz) {
        for (auto iu = 0; iu < 2; ++iu) {
          EXPECT_EQ(xr.q(ix, iy, iz, iu), ix * iy * (iz + 1) * (iu + 1));
        }
      }
    }
  }

  auto xr0 = xr.frame_2d(0, 0);
  EXPECT_EQ(xr0.x(), xr.x());
  EXPECT_EQ(xr0.y(), xr.y());

  for (auto ix = 0; ix < 6; ++ix) {
    for (auto iy = 0; iy < 8; ++iy) {
      EXPECT_EQ(xr0.q(ix, iy), ix * iy);
    }
  }

  auto xr1 = xr.frame_2d(1, 1);
  for (auto ix = 0; ix < 6; ++ix) {
    for (auto iy = 0; iy < 8; ++iy) {
      EXPECT_EQ(xr1.q(ix, iy), ix * iy * 2 * 2);
    }
  }
}

TEST(math_spline2d, spline2d) {
  auto xr = math::Frame2D(3, 3);
  ASSERT_EQ(xr.nx(), 3);
  ASSERT_EQ(xr.ny(), 3);

  for (auto ix = 0; ix < 6; ++ix) {
    xr.x(ix) = xr.y(ix) = ix * 0.1;
    for (auto iy = 0; iy < 6; ++iy) {
      xr.q(ix, iy) = std::sin(ix * 0.1);
    }
  }

  auto interpolator = math::Spline2D(xr, "c_spline");
  auto acc = gsl::Accelerator();
  for (auto ix = 0; ix < 6; ++ix) {
    for (auto iy = 0; iy < 6; ++iy) {
      EXPECT_EQ(interpolator.interpolate(ix * 0.1, iy * 0.1, xr),
                std::sin(ix * 0.1));
    }
  }
}
