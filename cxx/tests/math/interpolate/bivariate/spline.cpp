// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/interpolate/bivariate/spline.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cstdint>
#include <memory>

#include "pyinterp/math/interpolate/univariate/akima.hpp"
#include "pyinterp/math/interpolate/univariate/linear.hpp"

namespace pyinterp::math::interpolate::bivariate {

// Test bivariate spline with linear 1D interpolation on symmetric grid
TEST(Spline, LinearSymmetric) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(4, 4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(6);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(6);
  Eigen::Matrix<double, Eigen::Dynamic, 1> zp(6);

  xa << 0.0, 1.0, 2.0, 3.0;
  ya << 0.0, 1.0, 2.0, 3.0;
  za << 1.0, 1.1, 1.2, 1.3, 1.1, 1.2, 1.3, 1.4, 1.2, 1.3, 1.4, 1.5, 1.3, 1.4,
      1.5, 1.6;
  xp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0;
  yp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0;
  zp << 1.0, 1.1, 1.2, 1.3, 1.5, 1.6;

  auto spline = Spline<double>(std::make_unique<univariate::Linear<double>>());
  spline.prepare(xa, ya, za);
  auto z = spline(xp, yp);
  for (int64_t i = 0; i < z.size(); ++i) {
    EXPECT_NEAR(z(i), zp(i), 1.0e-12);
  }
}

// Test bivariate spline with linear 1D interpolation on asymmetric grid
TEST(Spline, LinearAsymmetric) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(4, 4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(12);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(12);
  Eigen::Matrix<double, Eigen::Dynamic, 1> zp(12);

  xa << 0.0, 1.0, 2.0, 3.0;
  ya << 0.0, 1.0, 2.0, 3.0;
  za << 1.0, 1.3, 1.5, 1.6, 1.1, 1.4, 1.6, 1.9, 1.2, 1.5, 1.7, 2.2, 1.4, 1.7,
      1.9, 2.3;
  xp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 1.3954, 1.6476, 0.824957, 2.41108,
      2.98619, 1.36485;
  yp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 0.265371, 2.13849, 1.62114, 1.22198,
      0.724681, 0.0596087;
  zp << 1.0, 1.2, 1.4, 1.55, 2.025, 2.3, 1.2191513, 1.7242442248, 1.5067237,
      1.626612, 1.6146423, 1.15436761;

  auto spline = Spline<double>(std::make_unique<univariate::Linear<double>>());
  spline.prepare(xa, ya, za);
  auto z = spline(xp, yp);
  for (int64_t i = 0; i < z.size(); ++i) {
    EXPECT_NEAR(z(i), zp(i), 1.0e-12);
  }
}

// Test bivariate spline with Akima 1D interpolation
TEST(Spline, AkimaSmooth) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(6, 6);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(6);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(6);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(5);

  // Create a smooth test function: f(x,y) = x^2 + y^2
  xa << 0.0, 1.0, 2.0, 3.0, 4.0, 5.0;
  ya << 0.0, 1.0, 2.0, 3.0, 4.0, 5.0;

  for (int64_t i = 0; i < 6; ++i) {
    for (int64_t j = 0; j < 6; ++j) {
      za(i, j) = xa(i) * xa(i) + ya(j) * ya(j);
    }
  }

  // Test points between grid points
  xp << 0.5, 1.5, 2.5, 3.5, 4.5;
  yp << 0.5, 1.5, 2.5, 3.5, 4.5;

  auto spline = Spline<double>(std::make_unique<univariate::Akima<double>>());
  spline.prepare(xa, ya, za);
  auto z = spline(xp, yp);

  // Check that interpolated values are reasonable for f(x,y) = x^2 + y^2
  for (int64_t i = 0; i < z.size(); ++i) {
    double expected = xp(i) * xp(i) + yp(i) * yp(i);
    // Akima should give good approximation for smooth functions
    EXPECT_NEAR(z(i), expected, 0.5);
  }
}

// Test on grid points (should return exact values)
TEST(Spline, ExactOnGridPoints) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(5, 5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(5);

  xa << 1.0, 2.0, 3.0, 4.0, 5.0;
  ya << 1.0, 2.0, 3.0, 4.0, 5.0;

  // Arbitrary values
  za << 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0, 3.0, 6.0, 9.0, 12.0,
      15.0, 4.0, 8.0, 12.0, 16.0, 20.0, 5.0, 10.0, 15.0, 20.0, 25.0;

  auto spline = Spline<double>(std::make_unique<univariate::Linear<double>>());
  spline.prepare(xa, ya, za);

  // Test exact values at grid points
  for (int64_t i = 0; i < 5; ++i) {
    for (int64_t j = 0; j < 5; ++j) {
      double z = spline(xa(i), ya(j));
      EXPECT_NEAR(z, za(i, j), 1.0e-12);
    }
  }
}

// Test with non-square grid
TEST(Spline, NonSquareGrid) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(6, 4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(6);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(5);

  xa << 0.0, 1.0, 2.0, 3.0, 4.0, 5.0;
  ya << 0.0, 2.0, 4.0, 6.0;

  // Simple linear function: f(x,y) = x + y
  for (int64_t i = 0; i < 6; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      za(i, j) = xa(i) + ya(j);
    }
  }

  xp << 0.5, 1.5, 2.5, 3.5, 4.5;
  yp << 1.0, 2.0, 3.0, 4.0, 5.0;

  auto spline = Spline<double>(std::make_unique<univariate::Linear<double>>());
  spline.prepare(xa, ya, za);
  auto z = spline(xp, yp);

  // For linear function with linear interpolation, should be exact
  for (int64_t i = 0; i < z.size(); ++i) {
    double expected = xp(i) + yp(i);
    EXPECT_NEAR(z(i), expected, 1.0e-12);
  }
}

// Test with column-major storage order
TEST(Spline, ColumnMajorStorage) {
  // Create column-major matrix explicitly
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> za(4,
                                                                            4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(3);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(3);

  xa << 0.0, 1.0, 2.0, 3.0;
  ya << 0.0, 1.0, 2.0, 3.0;
  za << 1.0, 1.1, 1.2, 1.3, 1.1, 1.2, 1.3, 1.4, 1.2, 1.3, 1.4, 1.5, 1.3, 1.4,
      1.5, 1.6;
  xp << 1.0, 1.5, 2.0;
  yp << 1.0, 1.5, 2.0;

  auto spline = Spline<double>(std::make_unique<univariate::Linear<double>>());
  spline.prepare(xa, ya, za);
  auto z = spline(xp, yp);

  // Expected values for linear interpolation
  Eigen::Matrix<double, Eigen::Dynamic, 1> zp(3);
  zp << 1.2, 1.3, 1.4;

  for (int64_t i = 0; i < z.size(); ++i) {
    EXPECT_NEAR(z(i), zp(i), 1.0e-12);
  }
}

// Test with row-major storage order
TEST(Spline, RowMajorStorage) {
  // Create row-major matrix explicitly
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> za(4,
                                                                            4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(3);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(3);

  xa << 0.0, 1.0, 2.0, 3.0;
  ya << 0.0, 1.0, 2.0, 3.0;
  za << 1.0, 1.1, 1.2, 1.3, 1.1, 1.2, 1.3, 1.4, 1.2, 1.3, 1.4, 1.5, 1.3, 1.4,
      1.5, 1.6;
  xp << 1.0, 1.5, 2.0;
  yp << 1.0, 1.5, 2.0;
  auto spline = Spline<double>(std::make_unique<univariate::Linear<double>>());
  spline.prepare(xa, ya, za);
  auto z = spline(xp, yp);

  // Expected values for linear interpolation
  Eigen::Matrix<double, Eigen::Dynamic, 1> zp(3);
  zp << 1.2, 1.3, 1.4;

  for (int64_t i = 0; i < z.size(); ++i) {
    EXPECT_NEAR(z(i), zp(i), 1.0e-12);
  }
}

// Test error handling for empty grid
TEST(Spline, EmptyGrid) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(0, 0);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(0);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(0);

  auto spline = Spline<double>(std::make_unique<univariate::Linear<double>>());
  EXPECT_TRUE(std::isnan(spline(1.0, 1.0)));
  spline.prepare(xa, ya, za);
  EXPECT_TRUE(std::isnan(spline(1.0, 1.0)));
}

// Test with larger grid to verify capacity management
TEST(Spline, LargeGrid) {
  const int64_t nx = 50;
  const int64_t ny = 40;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(nx, ny);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(nx);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(ny);

  for (int64_t i = 0; i < nx; ++i) {
    xa(i) = static_cast<double>(i);
  }
  for (int64_t j = 0; j < ny; ++j) {
    ya(j) = static_cast<double>(j);
  }

  // Fill grid with f(x,y) = 2*x + 3*y to have a clear linear pattern
  for (int64_t i = 0; i < nx; ++i) {
    for (int64_t j = 0; j < ny; ++j) {
      za(i, j) = 2.0 * xa(i) + 3.0 * ya(j);
    }
  }

  auto spline = Spline<double>(std::make_unique<univariate::Linear<double>>());
  spline.prepare(xa, ya, za);
  // Test multiple interpolations on a large grid
  // This verifies that internal storage resizes correctly
  double z1 = spline(10.0, 20.0);
  EXPECT_NEAR(z1, 2.0 * 10.0 + 3.0 * 20.0, 1.0e-10);

  double z2 = spline(10.5, 20.0);
  EXPECT_NEAR(z2, 2.0 * 10.5 + 3.0 * 20.0, 1.0e-10);

  double z3 = spline(10.0, 20.5);
  EXPECT_NEAR(z3, 2.0 * 10.0 + 3.0 * 20.5, 1.0e-10);

  double z4 = spline(25.5, 35.5);
  EXPECT_NEAR(z4, 2.0 * 25.5 + 3.0 * 35.5, 1.0e-10);
}

// Test batch interpolation with mixed coordinates
TEST(Spline, BatchInterpolation) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(5, 5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(10);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(10);

  xa << 0.0, 1.0, 2.0, 3.0, 4.0;
  ya << 0.0, 1.0, 2.0, 3.0, 4.0;

  // f(x,y) = 2*x + 3*y
  for (int64_t i = 0; i < 5; ++i) {
    for (int64_t j = 0; j < 5; ++j) {
      za(i, j) = 2.0 * xa(i) + 3.0 * ya(j);
    }
  }

  xp << 0.5, 1.5, 2.5, 3.5, 0.3, 1.7, 2.1, 3.9, 0.0, 4.0;
  yp << 0.5, 1.5, 2.5, 3.5, 0.7, 1.3, 2.9, 3.1, 0.0, 4.0;

  auto spline = Spline<double>(std::make_unique<univariate::Linear<double>>());
  spline.prepare(xa, ya, za);
  auto z = spline(xp, yp);
  // For linear function with linear interpolation, should be exact
  for (int64_t i = 0; i < z.size(); ++i) {
    double expected = 2.0 * xp(i) + 3.0 * yp(i);
    EXPECT_NEAR(z(i), expected, 1.0e-10);
  }
}

}  // namespace pyinterp::math::interpolate::bivariate
