// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/interpolate/geometric_cache.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "pyinterp/math/interpolate/geometric_cache_loader.hpp"

namespace pyinterp::math::interpolate::geometric {

// ==================== Cache Construction Tests ====================

TEST(GeometricCache2D, DefaultConstruction) {
  Cache2D<double> cache;

  // Initially cache has no valid domain
  EXPECT_FALSE(cache.has_domain());
  EXPECT_TRUE(cache.is_valid());  // No NaN values
}

TEST(GeometricCache2D, StaticProperties) {
  using CacheType = Cache2D<double>;

  EXPECT_EQ(CacheType::kNDim, 2);
  EXPECT_EQ(CacheType::kNumValues, 4);  // 2^2 = 4 corners
}

TEST(GeometricCache3D, StaticProperties) {
  using CacheType = Cache3D<double>;

  EXPECT_EQ(CacheType::kNDim, 3);
  EXPECT_EQ(CacheType::kNumValues, 8);  // 2^3 = 8 corners
}

TEST(GeometricCache4D, StaticProperties) {
  using CacheType = Cache4D<double>;

  EXPECT_EQ(CacheType::kNDim, 4);
  EXPECT_EQ(CacheType::kNumValues, 16);  // 2^4 = 16 corners
}

// ==================== Coordinate Access Tests ====================

TEST(GeometricCache2D, CoordinateAccess) {
  Cache2D<double> cache;

  cache.set_coords<0>(1.0, 2.0);
  cache.set_coords<1>(3.0, 4.0);

  EXPECT_DOUBLE_EQ(cache.coord_lower<0>(), 1.0);
  EXPECT_DOUBLE_EQ(cache.coord_upper<0>(), 2.0);
  EXPECT_DOUBLE_EQ(cache.coord_lower<1>(), 3.0);
  EXPECT_DOUBLE_EQ(cache.coord_upper<1>(), 4.0);
}

TEST(GeometricCache3D, MixedAxisTypes) {
  // Test 3D cache with int64_t for Z axis (temporal)
  Cache3D<double, int64_t> cache;

  cache.set_coords<0>(1.0, 2.0);
  cache.set_coords<1>(3.0, 4.0);
  cache.set_coords<2>(100LL, 200LL);

  EXPECT_DOUBLE_EQ(cache.coord_lower<0>(), 1.0);
  EXPECT_DOUBLE_EQ(cache.coord_upper<0>(), 2.0);
  EXPECT_EQ(cache.coord_lower<2>(), 100LL);
  EXPECT_EQ(cache.coord_upper<2>(), 200LL);
}

// ==================== Value Access Tests ====================

TEST(GeometricCache2D, ValueAccess) {
  Cache2D<double> cache;

  // Set values by linear index
  cache.value(0) = 10.0;  // (0, 0)
  cache.value(1) = 20.0;  // (0, 1)
  cache.value(2) = 30.0;  // (1, 0)
  cache.value(3) = 40.0;  // (1, 1)

  EXPECT_DOUBLE_EQ(cache.value(0), 10.0);
  EXPECT_DOUBLE_EQ(cache.value(1), 20.0);
  EXPECT_DOUBLE_EQ(cache.value(2), 30.0);
  EXPECT_DOUBLE_EQ(cache.value(3), 40.0);
}

TEST(GeometricCache2D, ValueAccessByIndices) {
  Cache2D<double> cache;

  // Set values by multi-dimensional indices
  cache.set_value_at(10.0, 0, 0);
  cache.set_value_at(20.0, 0, 1);
  cache.set_value_at(30.0, 1, 0);
  cache.set_value_at(40.0, 1, 1);

  EXPECT_DOUBLE_EQ(cache.value_at(0, 0), 10.0);
  EXPECT_DOUBLE_EQ(cache.value_at(0, 1), 20.0);
  EXPECT_DOUBLE_EQ(cache.value_at(1, 0), 30.0);
  EXPECT_DOUBLE_EQ(cache.value_at(1, 1), 40.0);
}

TEST(GeometricCache3D, ValueAccess) {
  Cache3D<double> cache;

  // Fill all 8 corners
  for (size_t i = 0; i < 8; ++i) {
    cache.value(i) = static_cast<double>(i * 10);
  }

  for (size_t i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(cache.value(i), static_cast<double>(i * 10));
  }
}

TEST(GeometricCache4D, ValueAccess) {
  Cache4D<double> cache;

  // Fill all 16 corners
  for (size_t i = 0; i < 16; ++i) {
    cache.value(i) = static_cast<double>(i * 5);
  }

  for (size_t i = 0; i < 16; ++i) {
    EXPECT_DOUBLE_EQ(cache.value(i), static_cast<double>(i * 5));
  }
}

// ==================== Domain/Contains Tests ====================

TEST(GeometricCache2D, DomainAndContains) {
  Cache2D<double> cache;

  // Initially no valid domain
  EXPECT_FALSE(cache.has_domain());
  EXPECT_FALSE(cache.contains(1.5, 3.5));

  // Set coordinates and finalize
  cache.set_coords<0>(1.0, 2.0);
  cache.set_coords<1>(3.0, 4.0);
  cache.finalize();

  EXPECT_TRUE(cache.has_domain());

  // Point inside
  EXPECT_TRUE(cache.contains(1.5, 3.5));

  // On boundaries
  EXPECT_TRUE(cache.contains(1.0, 3.0));
  EXPECT_TRUE(cache.contains(2.0, 4.0));
  EXPECT_TRUE(cache.contains(1.0, 4.0));
  EXPECT_TRUE(cache.contains(2.0, 3.0));

  // Outside
  EXPECT_FALSE(cache.contains(0.5, 3.5));
  EXPECT_FALSE(cache.contains(2.5, 3.5));
  EXPECT_FALSE(cache.contains(1.5, 2.5));
  EXPECT_FALSE(cache.contains(1.5, 4.5));
}

TEST(GeometricCache2D, Invalidate) {
  Cache2D<double> cache;

  cache.set_coords<0>(1.0, 2.0);
  cache.set_coords<1>(3.0, 4.0);
  cache.finalize();
  EXPECT_TRUE(cache.has_domain());

  cache.invalidate();
  EXPECT_FALSE(cache.has_domain());
}

// ==================== NaN Validation Tests ====================

TEST(GeometricCache2D, IsValidWithNaN) {
  Cache2D<double> cache;

  cache.value(0) = 1.0;
  cache.value(1) = 2.0;
  cache.value(2) = 3.0;
  cache.value(3) = 4.0;
  EXPECT_TRUE(cache.is_valid());

  cache.value(1) = std::nan("");
  EXPECT_FALSE(cache.is_valid());
}

TEST(GeometricCache3D, IsValidWithAllValid) {
  Cache3D<double> cache;

  for (size_t i = 0; i < 8; ++i) {
    cache.value(i) = static_cast<double>(i);
  }
  EXPECT_TRUE(cache.is_valid());
}

// ==================== Integration Tests ====================

// A simple 2D grid for testing cache loading
struct TestGrid2D {
  static constexpr size_t kNDim = 2;

  TestGrid2D() {
    // X axis: 0, 1, 2, 3, 4, 5
    x_coords_ = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    // Y axis: 0, 2, 4, 6, 8, 10
    y_coords_ = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0};

    // Values: f(x, y) = x * 10 + y
    values_.resize(6 * 6);
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        values_[i * 6 + j] =
            static_cast<double>(x_coords_[i] * 10 + y_coords_[j]);
      }
    }
  }

  template <size_t I>
  [[nodiscard]] auto construct_bounds_error_description(
      const double& coordinate) const -> std::string {
    return "Out of bounds";
  }

  template <size_t I>
  [[nodiscard]] auto axis() const noexcept -> const std::vector<double>& {
    if constexpr (I == 0) {
      return x_coords_;
    } else {
      return y_coords_;
    }
  }

  template <size_t I>
  [[nodiscard]] auto find_indexes(double coord, bool bounds_error) const
      -> std::optional<std::pair<size_t, size_t>> {
    const auto& coords = axis<I>();
    if (coord < coords.front() || coord > coords.back()) {
      return std::nullopt;
    }
    for (size_t i = 0; i < coords.size() - 1; ++i) {
      if (coord >= coords[i] && coord <= coords[i + 1]) {
        return std::make_pair(i, i + 1);
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] auto value(size_t i, size_t j) const noexcept -> double {
    return values_[i * 6 + j];
  }

  [[nodiscard]] auto coordinate_value(size_t dim, size_t idx) const noexcept
      -> double {
    if (dim == 0) {
      return x_coords_[idx];
    }
    return y_coords_[idx];
  }

 private:
  std::vector<double> x_coords_;
  std::vector<double> y_coords_;
  std::vector<double> values_;
};

// A mock grid interface that matches the expected API
struct MockGrid2D {
  static constexpr size_t kNDim = 2;

  MockGrid2D() : test_grid_() {}

  template <size_t I>
  [[nodiscard]] auto construct_bounds_error_description(
      const double& coordinate) const -> std::string {
    return test_grid_.construct_bounds_error_description<I>(coordinate);
  }

  struct MockAxis {
    const MockGrid2D* grid;
    size_t dim;

    [[nodiscard]] auto coordinate_value(size_t idx) const -> double {
      return grid->test_grid_.coordinate_value(dim, idx);
    }
  };

  template <size_t I>
  [[nodiscard]] auto axis() const noexcept -> MockAxis {
    return MockAxis{this, I};
  }

  template <size_t I>
  [[nodiscard]] auto find_indexes(double coord, bool bounds_error) const
      -> std::optional<std::pair<size_t, size_t>> {
    return test_grid_.find_indexes<I>(coord, bounds_error);
  }

  [[nodiscard]] auto value(size_t i, size_t j) const noexcept -> double {
    return test_grid_.value(i, j);
  }

 private:
  TestGrid2D test_grid_;
};

TEST(GeometricCacheLoader, UpdateCacheIfNeeded) {
  MockGrid2D grid;
  Cache2D<double> cache;

  // First load - should update
  auto result1 =
      update_cache_if_needed(cache, grid, std::make_tuple(1.5, 3.0), false);
  EXPECT_TRUE(result1.success);
  EXPECT_TRUE(result1.was_updated);
  EXPECT_FALSE(result1.error_message.has_value());

  // Verify cached coordinates
  EXPECT_DOUBLE_EQ(cache.coord_lower<0>(), 1.0);
  EXPECT_DOUBLE_EQ(cache.coord_upper<0>(), 2.0);
  EXPECT_DOUBLE_EQ(cache.coord_lower<1>(), 2.0);
  EXPECT_DOUBLE_EQ(cache.coord_upper<1>(), 4.0);

  // Second load with same cell - should NOT update (fast path)
  auto result2 =
      update_cache_if_needed(cache, grid, std::make_tuple(1.7, 3.5), false);
  EXPECT_TRUE(result2.success);
  EXPECT_FALSE(result2.was_updated);

  // Third load with different cell - should update
  auto result3 =
      update_cache_if_needed(cache, grid, std::make_tuple(3.5, 7.0), false);
  EXPECT_TRUE(result3.success);
  EXPECT_TRUE(result3.was_updated);

  // Verify new cached coordinates
  EXPECT_DOUBLE_EQ(cache.coord_lower<0>(), 3.0);
  EXPECT_DOUBLE_EQ(cache.coord_upper<0>(), 4.0);
  EXPECT_DOUBLE_EQ(cache.coord_lower<1>(), 6.0);
  EXPECT_DOUBLE_EQ(cache.coord_upper<1>(), 8.0);
}

TEST(GeometricCacheLoader, OutOfBounds) {
  MockGrid2D grid;
  Cache2D<double> cache;

  // Out of bounds - should fail
  auto result =
      update_cache_if_needed(cache, grid, std::make_tuple(-1.0, 3.0), false);
  EXPECT_FALSE(result.success);
  EXPECT_FALSE(cache.has_domain());

  // With bounds_error = true
  auto result2 =
      update_cache_if_needed(cache, grid, std::make_tuple(-1.0, 3.0), true);
  EXPECT_FALSE(result2.success);
  EXPECT_TRUE(result2.error_message.has_value());
}

TEST(GeometricCacheLoader, CornerValueMapping) {
  MockGrid2D grid;
  Cache2D<double> cache;

  // Load cell at (1.5, 3.0) - brackets are [1,2] x [2,4]
  auto result =
      update_cache_if_needed(cache, grid, std::make_tuple(1.5, 3.0), false);
  EXPECT_TRUE(result.success);

  // Expected values: f(x, y) = x * 10 + y
  // (1, 2) = 1*10 + 2 = 12  -> index 0
  // (1, 4) = 1*10 + 4 = 14  -> index 1
  // (2, 2) = 2*10 + 2 = 22  -> index 2
  // (2, 4) = 2*10 + 4 = 24  -> index 3
  EXPECT_DOUBLE_EQ(cache.value(0), 12.0);
  EXPECT_DOUBLE_EQ(cache.value(1), 14.0);
  EXPECT_DOUBLE_EQ(cache.value(2), 22.0);
  EXPECT_DOUBLE_EQ(cache.value(3), 24.0);
}

}  // namespace pyinterp::math::interpolate::geometric
