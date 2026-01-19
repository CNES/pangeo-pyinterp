// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/interpolate/window_cache_loader.hpp"

#include <gtest/gtest.h>
#include <sys/types.h>

#include "pyinterp/math/axis.hpp"

namespace pyinterp::math::interpolate {

// A simple 2D grid for testing purposes
struct Grid {
  // Number of dimensions
  static constexpr size_t kNDim = 2;

  // Out-of-bounds error message
  static constexpr const char* kOutOfBoundsMessage = "@Out of bounds@";

  // Build a grid with two axes and a simple matrix of values.
  //
  // The longitude axis is periodic (0 to 359 degrees) and the latitude axis is
  // non-periodic (-60 to 60 degrees).
  Grid() {
    std::get<0>(axes_) = math::Axis<double>(0, 359, 360, 1e-6, 360.0);
    std::get<1>(axes_) = math::Axis<double>(-60, 60, 121, 1e-6);
    matrix_.resize(360, 121);
    for (int64_t i = 0; i < 360; ++i) {
      for (int64_t j = 0; j < 121; ++j) {
        matrix_(i, j) = static_cast<uint8_t>((i + j) % 256);
      }
    }
  }

  // Construct an out-of-bounds error description for a given axis
  template <size_t I>
  [[nodiscard]] auto construct_bounds_error_description(
      const double& coordinate) const -> std::string {
    return kOutOfBoundsMessage;
  }

  // Return the axis at index I
  template <size_t I>
  [[nodiscard]] constexpr auto axis() const noexcept
      -> const math::Axis<double>& {
    return std::get<I>(axes_);
  }

  // Return the value at the given indices
  template <typename... Index>
  [[nodiscard]] auto value(Index&&... indices) const noexcept
      -> const uint8_t& {
    return matrix_(std::forward<Index>(indices)...);
  }

 private:
  // The axes of the grid
  std::tuple<math::Axis<double>, math::Axis<double>> axes_{};
  // The matrix of values
  Matrix<uint8_t> matrix_{};
};

// Update the cache if needed based on the query coordinates
TEST(CacheLoaderTest, LoadCacheGeneric) {
  Grid grid;
  InterpolationCache<double, double, double> cache(2, 2);

  // First load
  std::tuple<double, double> query_coords{359.5, 0.5};
  auto cached = update_cache_if_needed(cache, grid, query_coords,
                                       axis::Boundary::kWrap, true);
  EXPECT_TRUE(cached.success);
  EXPECT_TRUE(cached.was_updated);
  EXPECT_FALSE(cached.error_message.has_value());
  auto& lon = cache.coords<0>();
  EXPECT_EQ(lon[0], 358);
  EXPECT_EQ(lon[3], 361);

  // Second load with same coordinates
  cached = update_cache_if_needed(cache, grid, query_coords,
                                  axis::Boundary::kWrap, true);
  EXPECT_TRUE(cached.success);
  EXPECT_FALSE(cached.was_updated);
  EXPECT_FALSE(cached.error_message.has_value());

  // Third load with different coordinates inside the cached domain
  query_coords = std::make_tuple(359.51, .51);
  cached = update_cache_if_needed(cache, grid, query_coords,
                                  axis::Boundary::kWrap, true);
  EXPECT_TRUE(cached.success);
  EXPECT_FALSE(cached.was_updated);
  EXPECT_FALSE(cached.error_message.has_value());

  // Fourth load with out-of-bounds coordinates
  query_coords = std::make_tuple(0.0, -61.0);
  cached = update_cache_if_needed(cache, grid, query_coords,
                                  axis::Boundary::kWrap, true);
  EXPECT_FALSE(cached.success);
  EXPECT_TRUE(cached.was_updated);
  ASSERT_TRUE(cached.error_message.has_value());
  EXPECT_EQ(cached.error_message.value(), Grid::kOutOfBoundsMessage);

  // Fifth load with x coordinate within the periodic domain
  query_coords = std::make_tuple(-10.1, 0.1);
  cached = update_cache_if_needed(cache, grid, query_coords,
                                  axis::Boundary::kWrap, true);
  EXPECT_TRUE(cached.success);
  EXPECT_TRUE(cached.was_updated);
  EXPECT_FALSE(cached.error_message.has_value());
  lon = cache.coords<0>();
  EXPECT_NEAR(lon[0], -12, 1e-6);
  EXPECT_NEAR(lon[3], -9, 1e-6);

  // Sixth load with coordinates inside the cached domain (the cache domain
  // should have been shifted by the periodicity)
  cached = update_cache_if_needed(cache, grid, query_coords,
                                  axis::Boundary::kWrap, true);
  EXPECT_TRUE(cached.success);
  EXPECT_FALSE(cached.was_updated);
  EXPECT_FALSE(cached.error_message.has_value());
}

// Test kShrink boundary mode at lower boundary
TEST(CacheLoaderTest, ShrinkBoundaryLowerEdge) {
  Grid grid;
  InterpolationCache<double, double, double> cache(2, 2);

  // Query near the lower boundary of the latitude axis (-60)
  // With kShrink, the cache should contain fewer points on the lower side
  std::tuple<double, double> query_coords{180.0, -59.5};
  auto cached = update_cache_if_needed(cache, grid, query_coords,
                                       axis::Boundary::kShrink, false);

  EXPECT_TRUE(cached.success);
  EXPECT_TRUE(cached.was_updated);
  EXPECT_FALSE(cached.error_message.has_value());

  // Verify that the cache was loaded and contains valid data
  EXPECT_TRUE(cache.is_valid());
  EXPECT_TRUE(cache.has_domain());

  // Check that the latitude coordinates are shrunk (fewer points available)
  auto& lat = cache.coords<1>();
  // The cache should be shrunk since we're near the boundary
  // At -59.5, with window size 2, we'd normally want [-61, -60, -59, -58]
  // but -61 is out of bounds, so it should shrink to available points
  EXPECT_GT(lat.size(), 0);
  EXPECT_LE(lat.size(), 4);  // At most 4 points

  // The minimum latitude should be >= -60 (grid boundary)
  EXPECT_GE(*std::ranges::min_element(lat), -60.0);
}

// Test kShrink boundary mode at upper boundary
TEST(CacheLoaderTest, ShrinkBoundaryUpperEdge) {
  Grid grid;
  InterpolationCache<double, double, double> cache(2, 2);

  // Query near the upper boundary of the latitude axis (60)
  std::tuple<double, double> query_coords{180.0, 59.5};
  auto cached = update_cache_if_needed(cache, grid, query_coords,
                                       axis::Boundary::kShrink, false);

  EXPECT_TRUE(cached.success);
  EXPECT_TRUE(cached.was_updated);
  EXPECT_FALSE(cached.error_message.has_value());

  // Verify that the cache was loaded
  EXPECT_TRUE(cache.is_valid());
  EXPECT_TRUE(cache.has_domain());

  auto& lat = cache.coords<1>();
  // The cache should be shrunk since we're near the upper boundary
  EXPECT_GT(lat.size(), 0);
  EXPECT_LE(lat.size(), 4);

  // The maximum latitude should be <= 60 (grid boundary)
  EXPECT_LE(*std::ranges::max_element(lat), 60.0);
}

// Test kShrink boundary mode at corner (both axes at boundary)
TEST(CacheLoaderTest, ShrinkBoundaryCorner) {
  Grid grid;
  InterpolationCache<double, double, double> cache(2, 2);

  // Query at corner: near lower longitude boundary and upper latitude boundary
  // For periodic longitude (0-359), 0.5 should work fine
  // For latitude at 59.5, we're near upper boundary
  std::tuple<double, double> query_coords{0.5, 59.5};
  auto cached = update_cache_if_needed(cache, grid, query_coords,
                                       axis::Boundary::kShrink, false);

  EXPECT_TRUE(cached.success);
  EXPECT_TRUE(cached.was_updated);
  EXPECT_FALSE(cached.error_message.has_value());

  EXPECT_TRUE(cache.is_valid());
  EXPECT_TRUE(cache.has_domain());

  // Check latitude shrinking
  auto& lat = cache.coords<1>();
  EXPECT_GT(lat.size(), 0);
  EXPECT_LE(*std::ranges::max_element(lat), 60.0);
}

// Test kShrink boundary mode at exact boundary point
TEST(CacheLoaderTest, ShrinkBoundaryExactEdge) {
  Grid grid;
  InterpolationCache<double, double, double> cache(2, 2);

  // Query exactly at the lower boundary
  std::tuple<double, double> query_coords{180.0, -60.0};
  auto cached = update_cache_if_needed(cache, grid, query_coords,
                                       axis::Boundary::kShrink, false);

  EXPECT_TRUE(cached.success);
  EXPECT_TRUE(cached.was_updated);
  EXPECT_FALSE(cached.error_message.has_value());

  EXPECT_TRUE(cache.is_valid());
  EXPECT_TRUE(cache.has_domain());

  auto& lat = cache.coords<1>();
  EXPECT_GT(lat.size(), 0);
  // At the exact boundary, coordinates should be within grid bounds
  EXPECT_GE(*std::ranges::min_element(lat), -60.0);
  EXPECT_LE(*std::ranges::max_element(lat), 60.0);
}

// Test kShrink vs kUndef behavior - kShrink should succeed where kUndef fails
TEST(CacheLoaderTest, ShrinkVsUndefBehavior) {
  Grid grid;

  // Test with kUndef - should fail very close to the boundary
  // At -59.5, with half_window=2, we need points at [-61, -60, -59, -58]
  // but -61 is out of bounds
  InterpolationCache<double, double, double> cache_undef(2, 2);
  std::tuple<double, double> query_coords{180.0, -59.5};
  auto cached_undef = update_cache_if_needed(cache_undef, grid, query_coords,
                                             axis::Boundary::kUndef, false);

  // kUndef should fail because it can't get enough points near the boundary
  EXPECT_FALSE(cached_undef.success);

  // Test with kShrink - should succeed with shrunk cache
  InterpolationCache<double, double, double> cache_shrink(2, 2);
  auto cached_shrink = update_cache_if_needed(cache_shrink, grid, query_coords,
                                              axis::Boundary::kShrink, false);

  // kShrink should succeed by reducing the cache size
  EXPECT_TRUE(cached_shrink.success);
  EXPECT_TRUE(cache_shrink.is_valid());
  EXPECT_TRUE(cache_shrink.has_domain());
}

// Test kShrink with asymmetric cache windows
TEST(CacheLoaderTest, ShrinkBoundaryAsymmetricCache) {
  Grid grid;
  // Asymmetric cache: cubic in X (half_window=2), linear in Y (half_window=1)
  InterpolationCache<double, double, double> cache(2, 1);

  // Query near the lower Y boundary
  std::tuple<double, double> query_coords{180.0, -59.5};
  auto cached = update_cache_if_needed(cache, grid, query_coords,
                                       axis::Boundary::kShrink, false);

  EXPECT_TRUE(cached.success);
  EXPECT_TRUE(cached.was_updated);
  EXPECT_FALSE(cached.error_message.has_value());

  EXPECT_TRUE(cache.is_valid());

  // X should have 4 points (not affected by boundary)
  auto& lon = cache.coords<0>();
  EXPECT_EQ(lon.size(), 4);  // Full window for X

  // Y might be shrunk near boundary
  auto& lat = cache.coords<1>();
  EXPECT_GT(lat.size(), 0);
  EXPECT_LE(lat.size(), 2);  // At most 2 points (linear)
  EXPECT_GE(*std::ranges::min_element(lat), -60.0);
}

// Test kShrink maintains cache validity across multiple queries
TEST(CacheLoaderTest, ShrinkBoundaryMultipleQueries) {
  Grid grid;
  InterpolationCache<double, double, double> cache(2, 2);

  // First query near lower boundary
  std::tuple<double, double> query1{180.0, -59.0};
  auto cached1 = update_cache_if_needed(cache, grid, query1,
                                        axis::Boundary::kShrink, false);
  EXPECT_TRUE(cached1.success);
  EXPECT_TRUE(cached1.was_updated);

  // Second query in middle of grid (should reload with full window)
  std::tuple<double, double> query2{180.0, 0.0};
  auto cached2 = update_cache_if_needed(cache, grid, query2,
                                        axis::Boundary::kShrink, false);
  EXPECT_TRUE(cached2.success);
  EXPECT_TRUE(cached2.was_updated);  // Should update since domain changed

  auto& lat = cache.coords<1>();
  EXPECT_EQ(lat.size(), 4);  // Full window in middle of grid

  // Third query back near boundary
  std::tuple<double, double> query3{180.0, 59.0};
  auto cached3 = update_cache_if_needed(cache, grid, query3,
                                        axis::Boundary::kShrink, false);
  EXPECT_TRUE(cached3.success);
  EXPECT_TRUE(cached3.was_updated);
}

// Test kShrink with bounds_error = true
TEST(CacheLoaderTest, ShrinkBoundaryWithBoundsError) {
  Grid grid;
  InterpolationCache<double, double, double> cache(2, 2);

  // Query near boundary with bounds_error enabled
  // kShrink should still succeed (it's within grid, just shrinks the window)
  std::tuple<double, double> query_coords{180.0, -59.5};
  auto cached = update_cache_if_needed(cache, grid, query_coords,
                                       axis::Boundary::kShrink, true);

  EXPECT_TRUE(cached.success);
  EXPECT_FALSE(cached.error_message.has_value());

  // Query outside grid boundary with bounds_error enabled
  std::tuple<double, double> query_outside{180.0, -61.0};
  auto cached_outside = update_cache_if_needed(cache, grid, query_outside,
                                               axis::Boundary::kShrink, true);

  EXPECT_FALSE(cached_outside.success);
  EXPECT_TRUE(cached_outside.error_message.has_value());
  EXPECT_EQ(cached_outside.error_message.value(), Grid::kOutOfBoundsMessage);
}

}  // namespace pyinterp::math::interpolate
