// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/interpolate/cache.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <utility>

#include "pyinterp/math/interpolate/cache.hpp"

namespace pyinterp::math::interpolate {

// Helper to ensure correct macro expansion for multi-index operator[]
// (prevents issues with comma operator in EXPECT macros)
auto expect_float_eq(float a, float b) -> void { EXPECT_FLOAT_EQ(a, b); }

TEST(IndependentCache1D, Construction) {
  // 1D cache: only X dimension, Y is ignored
  InterpolationCache<float, double> cache_linear(1, 1);
  EXPECT_EQ(cache_linear.x_half_window(), 1);
  EXPECT_EQ(cache_linear.x_points(), 2);

  InterpolationCache<float, double> cache_cubic(2, 2);
  EXPECT_EQ(cache_cubic.x_half_window(), 2);
  EXPECT_EQ(cache_cubic.x_points(), 4);
}

TEST(IndependentCache1D, CoordinateAccess) {
  InterpolationCache<float, double> cache(2, 2);

  cache.set_coord<0>(0, 1.0);
  cache.set_coord<0>(1, 2.0);
  cache.set_coord<0>(2, 3.0);
  cache.set_coord<0>(3, 4.0);

  EXPECT_DOUBLE_EQ(cache.coord<0>(0), 1.0);
  EXPECT_DOUBLE_EQ(cache.coord<0>(3), 4.0);
}

// ==================== 2D Cache Tests ====================

TEST(IndependentCache2D, SymmetricWindow) {
  // Same window size for both dimensions (like before)
  InterpolationCache<float, double, double> cache(2, 2);

  EXPECT_EQ(cache.x_half_window(), 2);
  EXPECT_EQ(cache.y_half_window(), 2);
  EXPECT_EQ(cache.x_points(), 4);
  EXPECT_EQ(cache.y_points(), 4);
  EXPECT_EQ(cache.values_flat().size(), 16);  // 4×4
}

TEST(IndependentCache2D, AsymmetricWindow) {
  // Different window sizes: cubic in X, linear in Y
  InterpolationCache<float, double, double> cache(2, 1);

  EXPECT_EQ(cache.x_half_window(), 2);
  EXPECT_EQ(cache.y_half_window(), 1);
  EXPECT_EQ(cache.x_points(), 4);
  EXPECT_EQ(cache.y_points(), 2);
  EXPECT_EQ(cache.values_flat().size(), 8);  // 4×2
}

TEST(IndependentCache2D, ReverseAsymmetric) {
  // Different window sizes: linear in X, cubic in Y
  InterpolationCache<float, double, double> cache(1, 2);

  EXPECT_EQ(cache.x_half_window(), 1);
  EXPECT_EQ(cache.y_half_window(), 2);
  EXPECT_EQ(cache.x_points(), 2);
  EXPECT_EQ(cache.y_points(), 4);
  EXPECT_EQ(cache.values_flat().size(), 8);  // 2×4
}

TEST(IndependentCache2D, ValueAccessAsymmetric) {
  // 4×2 cache (cubic X, linear Y)
  InterpolationCache<float, double, double> cache(2, 1);

  // Set all values
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      cache[i, j] = static_cast<float>(i * 10 + j);
    }
  }

  // Verify
  expect_float_eq(cache[0, 0], 0.0f);
  expect_float_eq(cache[0, 1], 1.0f);
  expect_float_eq(cache[3, 0], 30.0f);
  expect_float_eq(cache[3, 1], 31.0f);
}

TEST(IndependentCache2D, MatrixViewAsymmetric) {
  // 4×2 cache
  InterpolationCache<float, double, double> cache(2, 1);

  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      cache[i, j] = static_cast<float>(i + j);
    }
  }

  auto mat = cache.matrix();

  EXPECT_EQ(mat.rows(), 4);
  EXPECT_EQ(mat.cols(), 2);
  EXPECT_FLOAT_EQ(mat(2, 1), 3.0f);  // i=2, j=1: 2+1=3
}

TEST(IndependentCache2D, DomainTrackingAsymmetric) {
  // Cubic X, Linear Y
  InterpolationCache<float, double, double> cache(2, 1);

  // Set X coordinates (4 points)
  for (size_t i = 0; i < 4; ++i) {
    cache.set_coord<0>(i, static_cast<double>(i) * 10.0);
  }

  // Set Y coordinates (2 points)
  for (size_t j = 0; j < 2; ++j) {
    cache.set_coord<1>(j, static_cast<double>(j) * 100.0);
  }

  cache.finalize({std::make_pair(1, 2), std::make_pair(0, 1)});

  EXPECT_TRUE(cache.has_domain());

  // X domain: window=2, points [0,10,20,30], domain=[10,20]
  // Y domain: window=1, points [0,100], domain=[0,100]
  EXPECT_TRUE(cache.contains(15.0, 50.0));
  EXPECT_TRUE(cache.contains(10.0, 0.0));
  EXPECT_TRUE(cache.contains(20.0, 100.0));
  EXPECT_FALSE(cache.contains(5.0, 50.0));    // X out of domain
  EXPECT_FALSE(cache.contains(15.0, 150.0));  // Y out of domain
}

// ==================== 3D Cache Tests ====================

TEST(IndependentCache3D, SymmetricXY) {
  // X and Y symmetric, Z always 2 points
  InterpolationCache<float, double, double, double> cache(2, 2);

  EXPECT_EQ(cache.x_points(), 4);
  EXPECT_EQ(cache.y_points(), 4);
  EXPECT_EQ(cache.points_per_dim(2), 2);      // Z always 2
  EXPECT_EQ(cache.values_flat().size(), 32);  // 4×4×2
}

TEST(IndependentCache3D, AsymmetricXY) {
  // X cubic, Y linear, Z always 2
  InterpolationCache<float, double, double, double> cache(2, 1);

  EXPECT_EQ(cache.x_points(), 4);
  EXPECT_EQ(cache.y_points(), 2);
  EXPECT_EQ(cache.points_per_dim(2), 2);      // Z always 2
  EXPECT_EQ(cache.values_flat().size(), 16);  // 4×2×2
}

TEST(IndependentCache3D, MatrixSliceAsymmetric) {
  // 4×2×2 cache (cubic X, linear Y, cubic Z)
  InterpolationCache<float, double, double, double> cache(2, 1);

  // Fill z=1 slice
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      cache[i, j, 1] = static_cast<float>(i * 10 + j);
    }
  }

  auto slice = cache.matrix(1);
  EXPECT_EQ(slice.rows(), 4);  // X dimension
  EXPECT_EQ(slice.cols(), 2);  // Y dimension
  EXPECT_FLOAT_EQ(slice(1, 0), 10.0f);
  EXPECT_FLOAT_EQ(slice(3, 1), 31.0f);
}

TEST(IndependentCache3D, DomainTracking) {
  InterpolationCache<float, double, double, double> cache(2, 1);

  // Set coordinates
  for (size_t i = 0; i < 4; ++i) {
    cache.set_coord<0>(i, static_cast<double>(i) * 10.0);  // X: [0,10,20,30]
  }
  for (size_t j = 0; j < 2; ++j) {
    cache.set_coord<1>(j, static_cast<double>(j) * 100.0);  // Y: [0,100]
  }
  for (size_t k = 0; k < 2; ++k) {
    cache.set_coord<2>(k, static_cast<double>(k) * 5.0);  // Z: [0,5]
  }

  cache.finalize({std::make_pair(1, 2),    // X: [10,20]
                  std::make_pair(0, 1),    // Y: [0,100]
                  std::make_pair(0, 1)});  // Z: [0,5]

  EXPECT_TRUE(cache.has_domain());

  // X domain: [10, 20], Y domain: [0, 100], Z domain: [0, 5]
  EXPECT_TRUE(cache.contains(15.0, 50.0, 2.5));
  EXPECT_FALSE(cache.contains(5.0, 50.0, 2.5));    // X out
  EXPECT_FALSE(cache.contains(15.0, 150.0, 2.5));  // Y out
  EXPECT_FALSE(cache.contains(15.0, 50.0, 7.0));   // Z out
}

// ==================== 4D Cache Tests ====================

TEST(IndependentCache4D, AsymmetricXY) {
  // X cubic, Y linear, Z and U always 2
  InterpolationCache<float, double, double, double, double> cache(2, 1);

  EXPECT_EQ(cache.x_points(), 4);
  EXPECT_EQ(cache.y_points(), 2);
  EXPECT_EQ(cache.points_per_dim(2), 2);      // Z always 2
  EXPECT_EQ(cache.points_per_dim(3), 2);      // U always 2
  EXPECT_EQ(cache.values_flat().size(), 32);  // 4×2×2×2
}

TEST(IndependentCache4D, MatrixSliceAsymmetric) {
  InterpolationCache<float, double, double, double, double> cache(2, 1);

  // Fill (z=0, u=1) slice
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      cache[i, j, 0, 1] = static_cast<float>(i * 10 + j);
    }
  }

  auto slice = cache.matrix(0, 1);
  EXPECT_EQ(slice.rows(), 4);  // X
  EXPECT_EQ(slice.cols(), 2);  // Y
  EXPECT_FLOAT_EQ(slice(2, 1), 21.0f);
}

// ==================== Different Window Combinations ====================

TEST(WindowCombinations, LinearLinear) {
  // Linear in both X and Y
  InterpolationCache<float, double, double> cache(1, 1);

  EXPECT_EQ(cache.x_points(), 2);
  EXPECT_EQ(cache.y_points(), 2);
  EXPECT_EQ(cache.values_flat().size(), 4);  // 2×2

  cache[0, 0] = 1.0f;
  cache[0, 1] = 2.0f;
  cache[1, 0] = 3.0f;
  cache[1, 1] = 4.0f;

  auto mat = cache.matrix();
  EXPECT_FLOAT_EQ(mat.sum(), 10.0f);
}

TEST(WindowCombinations, CubicLinear) {
  // Cubic X, Linear Y
  InterpolationCache<float, double, double> cache(2, 1);

  EXPECT_EQ(cache.x_points(), 4);
  EXPECT_EQ(cache.y_points(), 2);
  EXPECT_EQ(cache.values_flat().size(), 8);  // 4×2
}

TEST(WindowCombinations, LinearCubic) {
  // Linear X, Cubic Y
  InterpolationCache<float, double, double> cache(1, 2);

  EXPECT_EQ(cache.x_points(), 2);
  EXPECT_EQ(cache.y_points(), 4);
  EXPECT_EQ(cache.values_flat().size(), 8);  // 2×4
}

TEST(WindowCombinations, CubicCubic) {
  // Cubic in both
  InterpolationCache<float, double, double> cache(2, 2);

  EXPECT_EQ(cache.x_points(), 4);
  EXPECT_EQ(cache.y_points(), 4);
  EXPECT_EQ(cache.values_flat().size(), 16);  // 4×4
}

TEST(WindowCombinations, QuinticLinear) {
  // Quintic X, Linear Y
  InterpolationCache<float, double, double> cache(3, 1);

  EXPECT_EQ(cache.x_points(), 6);
  EXPECT_EQ(cache.y_points(), 2);
  EXPECT_EQ(cache.values_flat().size(), 12);  // 6×2
}

TEST(WindowCombinations, LinearQuintic) {
  // Linear X, Quintic Y
  InterpolationCache<float, double, double> cache(1, 3);

  EXPECT_EQ(cache.x_points(), 2);
  EXPECT_EQ(cache.y_points(), 6);
  EXPECT_EQ(cache.values_flat().size(), 12);  // 2×6
}

TEST(WindowCombinations, QuinticCubic) {
  // Quintic X, Cubic Y
  InterpolationCache<float, double, double> cache(3, 2);

  EXPECT_EQ(cache.x_points(), 6);
  EXPECT_EQ(cache.y_points(), 4);
  EXPECT_EQ(cache.values_flat().size(), 24);  // 6×4
}

// ==================== Stride Tests ====================

TEST(Strides, AsymmetricLayout) {
  // 4×2 cache (cubic X, linear Y)
  InterpolationCache<float, double, double> cache(2, 1);

  // Fill with sequential values
  auto& flat = cache.values_flat();
  for (size_t i = 0; i < flat.size(); ++i) {
    flat[i] = static_cast<float>(i);
  }

  // Strides for 4×2: [2, 1]
  // index = i*2 + j*1

  expect_float_eq(cache[0, 0], 0.0f);  // 0*2 + 0*1 = 0
  expect_float_eq(cache[0, 1], 1.0f);  // 0*2 + 1*1 = 1
  expect_float_eq(cache[1, 0], 2.0f);  // 1*2 + 0*1 = 2
  expect_float_eq(cache[1, 1], 3.0f);  // 1*2 + 1*1 = 3
  expect_float_eq(cache[3, 0], 6.0f);  // 3*2 + 0*1 = 6
  expect_float_eq(cache[3, 1], 7.0f);  // 3*2 + 1*1 = 7
}

TEST(Strides, 3DAsymmetric) {
  // 4×2×2 cache
  InterpolationCache<float, double, double, double> cache(2, 1);

  auto& flat = cache.values_flat();
  for (size_t i = 0; i < flat.size(); ++i) {
    flat[i] = static_cast<float>(i);
  }

  // Strides for 4×2×2: [4, 2, 1]
  // index = i*4 + j*2 + k*1

  expect_float_eq(cache[0, 0, 0], 0.0f);   // 0*4 + 0*2 + 0*1 = 0
  expect_float_eq(cache[0, 0, 1], 1.0f);   // 0*4 + 0*2 + 1*1 = 1
  expect_float_eq(cache[0, 1, 0], 2.0f);   // 0*4 + 1*2 + 0*1 = 2
  expect_float_eq(cache[1, 0, 0], 4.0f);   // 1*4 + 0*2 + 0*1 = 4
  expect_float_eq(cache[2, 1, 1], 11.0f);  // 2*4 + 1*2 + 1*1 = 11
}

// ==================== Matrix Operations ====================

TEST(MatrixOps, AsymmetricOperations) {
  // 4×2 cache
  InterpolationCache<float, double, double> cache(2, 1);

  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      cache[i, j] = static_cast<float>((i + 1) * (j + 1));
    }
  }

  auto mat = cache.matrix();

  // Matrix:
  // [1, 2]
  // [2, 4]
  // [3, 6]
  // [4, 8]

  EXPECT_FLOAT_EQ(mat.sum(), 30.0f);   // 1+2+2+4+3+6+4+8
  EXPECT_FLOAT_EQ(mat.mean(), 3.75f);  // 30/8

  auto row2 = mat.row(2);
  EXPECT_FLOAT_EQ(row2.sum(), 9.0f);  // 3+6

  auto col1 = mat.col(1);
  EXPECT_FLOAT_EQ(col1.sum(), 20.0f);  // 2+4+6+8
}

// ==================== Resize Tests ====================

TEST(Resize, 1DCache) {
  InterpolationCache<float, double> cache(2, 2);
  EXPECT_EQ(cache.x_points(), 4);

  // Resize to different size
  cache.resize({6});
  EXPECT_EQ(cache.x_points(), 6);
  EXPECT_EQ(cache.points_per_dim(0), 6);
  EXPECT_EQ(cache.values_flat().size(), 6);

  // Verify coordinates are resized
  for (size_t i = 0; i < 6; ++i) {
    cache.set_coord<0>(i, static_cast<double>(i) * 10.0);
  }
  EXPECT_DOUBLE_EQ(cache.coord<0>(5), 50.0);

  // Verify values can be set
  for (size_t i = 0; i < 6; ++i) {
    cache[i] = static_cast<float>(i);
  }
  expect_float_eq(cache[5], 5.0f);
}

TEST(Resize, 2DCache) {
  InterpolationCache<float, double, double> cache(2, 1);
  EXPECT_EQ(cache.x_points(), 4);
  EXPECT_EQ(cache.y_points(), 2);
  EXPECT_EQ(cache.values_flat().size(), 8);

  // Resize to different dimensions
  cache.resize({3, 5});
  EXPECT_EQ(cache.x_points(), 3);
  EXPECT_EQ(cache.y_points(), 5);
  EXPECT_EQ(cache.points_per_dim(0), 3);
  EXPECT_EQ(cache.points_per_dim(1), 5);
  EXPECT_EQ(cache.values_flat().size(), 15);  // 3×5

  // Verify coordinates are resized
  for (size_t i = 0; i < 3; ++i) {
    cache.set_coord<0>(i, static_cast<double>(i) * 10.0);
  }
  for (size_t j = 0; j < 5; ++j) {
    cache.set_coord<1>(j, static_cast<double>(j) * 100.0);
  }

  EXPECT_DOUBLE_EQ(cache.coord<0>(2), 20.0);
  EXPECT_DOUBLE_EQ(cache.coord<1>(4), 400.0);

  // Verify values can be set and accessed
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      cache[i, j] = static_cast<float>(i * 10 + j);
    }
  }

  expect_float_eq(cache[0, 0], 0.0f);
  expect_float_eq(cache[2, 4], 24.0f);

  // Verify matrix view
  auto mat = cache.matrix();
  EXPECT_EQ(mat.rows(), 3);
  EXPECT_EQ(mat.cols(), 5);
  EXPECT_FLOAT_EQ(mat(1, 3), 13.0f);
}

TEST(Resize, 3DCache) {
  InterpolationCache<float, double, double, double> cache(2, 1);
  EXPECT_EQ(cache.values_flat().size(), 16);  // 4×2×2

  // Resize to different dimensions
  cache.resize({5, 3, 4});
  EXPECT_EQ(cache.x_points(), 5);
  EXPECT_EQ(cache.y_points(), 3);
  EXPECT_EQ(cache.points_per_dim(0), 5);
  EXPECT_EQ(cache.points_per_dim(1), 3);
  EXPECT_EQ(cache.points_per_dim(2), 4);
  EXPECT_EQ(cache.values_flat().size(), 60);  // 5×3×4

  // Verify values can be set and accessed
  cache[2, 1, 3] = 42.0f;
  expect_float_eq(cache[2, 1, 3], 42.0f);

  // Verify matrix slice
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      cache[i, j, 2] = static_cast<float>(i + j);
    }
  }

  auto mat = cache.matrix(2);
  EXPECT_EQ(mat.rows(), 5);
  EXPECT_EQ(mat.cols(), 3);
  EXPECT_FLOAT_EQ(mat(3, 1), 4.0f);  // 3+1=4
}

TEST(Resize, 4DCache) {
  InterpolationCache<float, double, double, double, double> cache(2, 1);
  EXPECT_EQ(cache.values_flat().size(), 32);  // 4×2×2×2

  // Resize to different dimensions
  cache.resize({2, 3, 4, 5});
  EXPECT_EQ(cache.x_points(), 2);
  EXPECT_EQ(cache.y_points(), 3);
  EXPECT_EQ(cache.points_per_dim(2), 4);
  EXPECT_EQ(cache.points_per_dim(3), 5);
  EXPECT_EQ(cache.values_flat().size(), 120);  // 2×3×4×5

  // Verify values can be set and accessed
  cache[1, 2, 3, 4] = 99.0f;
  expect_float_eq(cache[1, 2, 3, 4], 99.0f);
}

TEST(Resize, MultipleResizes) {
  InterpolationCache<float, double, double> cache(2, 1);

  // First resize
  cache.resize({3, 3});
  EXPECT_EQ(cache.values_flat().size(), 9);

  // Fill with values
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      cache[i, j] = static_cast<float>(i * 3 + j);
    }
  }
  expect_float_eq(cache[2, 2], 8.0f);

  // Second resize - smaller
  cache.resize({2, 2});
  EXPECT_EQ(cache.values_flat().size(), 4);

  // Third resize - larger
  cache.resize({5, 4});
  EXPECT_EQ(cache.values_flat().size(), 20);
  EXPECT_EQ(cache.x_points(), 5);
  EXPECT_EQ(cache.y_points(), 4);

  // Verify new size works
  cache[4, 3] = 123.0f;
  expect_float_eq(cache[4, 3], 123.0f);
}

TEST(Resize, PreservesStrides) {
  InterpolationCache<float, double, double, double> cache(1, 1);

  cache.resize({3, 4, 2});

  // Fill with sequential values
  auto& flat = cache.values_flat();
  for (size_t i = 0; i < flat.size(); ++i) {
    flat[i] = static_cast<float>(i);
  }

  // Verify strides are correct (3×4×2 layout)
  // Strides should be [8, 2, 1]
  expect_float_eq(cache[0, 0, 0], 0.0f);   // 0*8 + 0*2 + 0*1 = 0
  expect_float_eq(cache[0, 0, 1], 1.0f);   // 0*8 + 0*2 + 1*1 = 1
  expect_float_eq(cache[0, 1, 0], 2.0f);   // 0*8 + 1*2 + 0*1 = 2
  expect_float_eq(cache[1, 0, 0], 8.0f);   // 1*8 + 0*2 + 0*1 = 8
  expect_float_eq(cache[2, 3, 1], 23.0f);  // 2*8 + 3*2 + 1*1 = 23
}

// ==================== Performance Tests ====================

TEST(Performance, AsymmetricCache) {
  InterpolationCache<float, double, double> cache(2, 1);

  constexpr int iterations = 100000;
  auto start = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < iterations; ++iter) {
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        cache[i, j] = static_cast<float>(i * j);
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  auto time_ns = static_cast<int>(duration / static_cast<double>(iterations));
  RecordProperty("AsymmetricCacheTimeNS", time_ns);
}

}  // namespace pyinterp::math::interpolate
