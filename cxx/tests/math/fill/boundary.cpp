// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/fill/boundary.hpp"

#include <gtest/gtest.h>

namespace pyinterp::fill {

// detail::periodic_index tests
TEST(BoundaryDetailTest, PeriodicIndexBasic) {
  using detail::periodic_index;

  // In-range indices unchanged
  EXPECT_EQ(periodic_index(0, 10), 0);
  EXPECT_EQ(periodic_index(5, 10), 5);
  EXPECT_EQ(periodic_index(9, 10), 9);

  // Wrapping at boundaries (examples from header docs)
  EXPECT_EQ(periodic_index(-1, 10), 9);
  EXPECT_EQ(periodic_index(10, 10), 0);
  EXPECT_EQ(periodic_index(11, 10), 1);
}

// detail::reflective_index tests
TEST(BoundaryDetailTest, ReflectiveIndexBasic) {
  using detail::reflective_index;

  // In-range indices unchanged
  EXPECT_EQ(reflective_index(0, 10), 0);
  EXPECT_EQ(reflective_index(5, 10), 5);
  EXPECT_EQ(reflective_index(9, 10), 9);

  // Reflection at boundaries
  EXPECT_EQ(reflective_index(-1, 10), 1);
  EXPECT_EQ(reflective_index(-2, 10), 2);
  EXPECT_EQ(reflective_index(10, 10), 8);
  EXPECT_EQ(reflective_index(11, 10), 7);
  EXPECT_EQ(reflective_index(19, 10), 1);
}

// Neighbors compile-time configuration tests
TEST(BoundaryNeighborsTest, PeriodicXReflectiveY) {
  // Grid 5x4
  constexpr int64_t nx = 5;
  constexpr int64_t ny = 4;
  Neighbors<true, false> nbr(nx, ny);

  // Interior cell
  nbr.update_x(2);
  EXPECT_EQ(nbr.ix0, 1);
  EXPECT_EQ(nbr.ix1, 3);

  nbr.update_y(2);
  EXPECT_EQ(nbr.iy0, 1);
  EXPECT_EQ(nbr.iy1, 3);

  // Left edge (x periodic wraps)
  nbr.update_x(0);
  EXPECT_EQ(nbr.ix0, 4);
  EXPECT_EQ(nbr.ix1, 1);

  // Right edge (x periodic wraps)
  nbr.update_x(4);
  EXPECT_EQ(nbr.ix0, 3);
  EXPECT_EQ(nbr.ix1, 0);

  // Bottom edge (y reflective)
  nbr.update_y(0);
  EXPECT_EQ(nbr.iy0, 1);
  EXPECT_EQ(nbr.iy1, 1);

  // Top edge (y reflective)
  nbr.update_y(3);
  EXPECT_EQ(nbr.iy0, 2);
  EXPECT_EQ(nbr.iy1, 2);
}

TEST(BoundaryNeighborsTest, ReflectiveXPeriodicY) {
  // Grid 5x4
  constexpr int64_t nx = 5;
  constexpr int64_t ny = 4;
  Neighbors<false, true> nbr(nx, ny);

  // Interior cell
  nbr.update_x(2);
  EXPECT_EQ(nbr.ix0, 1);
  EXPECT_EQ(nbr.ix1, 3);

  nbr.update_y(2);
  EXPECT_EQ(nbr.iy0, 1);
  EXPECT_EQ(nbr.iy1, 3);

  // Left edge (x reflective)
  nbr.update_x(0);
  EXPECT_EQ(nbr.ix0, 1);
  EXPECT_EQ(nbr.ix1, 1);

  // Right edge (x reflective)
  nbr.update_x(4);
  EXPECT_EQ(nbr.ix0, 3);
  EXPECT_EQ(nbr.ix1, 3);

  // Bottom edge (y periodic wraps)
  nbr.update_y(0);
  EXPECT_EQ(nbr.iy0, 3);
  EXPECT_EQ(nbr.iy1, 1);

  // Top edge (y periodic wraps)
  nbr.update_y(3);
  EXPECT_EQ(nbr.iy0, 2);
  EXPECT_EQ(nbr.iy1, 0);
}

TEST(BoundaryNeighborsTest, PeriodicXPeriodicY) {
  // Grid 3x3 to simplify checks
  constexpr int64_t n = 3;
  Neighbors<true, true> nbr(n, n);

  // Center
  nbr.update_x(1);
  nbr.update_y(1);
  EXPECT_EQ(nbr.ix0, 0);
  EXPECT_EQ(nbr.ix1, 2);
  EXPECT_EQ(nbr.iy0, 0);
  EXPECT_EQ(nbr.iy1, 2);

  // Corner (0,0) wraps around
  nbr.update_x(0);
  nbr.update_y(0);
  EXPECT_EQ(nbr.ix0, 2);
  EXPECT_EQ(nbr.ix1, 1);
  EXPECT_EQ(nbr.iy0, 2);
  EXPECT_EQ(nbr.iy1, 1);
}

TEST(BoundaryNeighborsTest, ReflectiveXReflectiveY) {
  // Grid 3x3
  constexpr int64_t n = 3;
  Neighbors<false, false> nbr(n, n);

  // Center
  nbr.update_x(1);
  nbr.update_y(1);
  EXPECT_EQ(nbr.ix0, 0);
  EXPECT_EQ(nbr.ix1, 2);
  EXPECT_EQ(nbr.iy0, 0);
  EXPECT_EQ(nbr.iy1, 2);

  // Corner (0,0) reflects
  nbr.update_x(0);
  nbr.update_y(0);
  EXPECT_EQ(nbr.ix0, 1);
  EXPECT_EQ(nbr.ix1, 1);
  EXPECT_EQ(nbr.iy0, 1);
  EXPECT_EQ(nbr.iy1, 1);

  // Corner (2,2) reflects
  nbr.update_x(2);
  nbr.update_y(2);
  EXPECT_EQ(nbr.ix0, 1);
  EXPECT_EQ(nbr.ix1, 1);
  EXPECT_EQ(nbr.iy0, 1);
  EXPECT_EQ(nbr.iy1, 1);
}

// DynamicNeighbors runtime configuration tests
TEST(BoundaryDynamicNeighborsTest, MatchesCompileTimeVariants) {
  constexpr int64_t nx = 5;
  constexpr int64_t ny = 4;

  // Periodic X, Reflective Y
  DynamicNeighbors d1(nx, ny, /*x_periodic=*/true, /*y_periodic=*/false);
  d1.update_x(0);
  d1.update_y(0);
  EXPECT_EQ(d1.ix0, 4);
  EXPECT_EQ(d1.ix1, 1);
  EXPECT_EQ(d1.iy0, 1);
  EXPECT_EQ(d1.iy1, 1);

  // Reflective X, Periodic Y
  DynamicNeighbors d2(nx, ny, /*x_periodic=*/false, /*y_periodic=*/true);
  d2.update_x(4);
  d2.update_y(3);
  EXPECT_EQ(d2.ix0, 3);
  EXPECT_EQ(d2.ix1, 3);
  EXPECT_EQ(d2.iy0, 2);
  EXPECT_EQ(d2.iy1, 0);

  // Periodic X, Periodic Y
  DynamicNeighbors d3(nx, ny, /*x_periodic=*/true, /*y_periodic=*/true);
  d3.update_x(0);
  d3.update_y(3);
  EXPECT_EQ(d3.ix0, 4);
  EXPECT_EQ(d3.ix1, 1);
  EXPECT_EQ(d3.iy0, 2);
  EXPECT_EQ(d3.iy1, 0);
}

// Simple constexpr checks (compiles if constexpr-evaluable)
TEST(BoundaryConstexprTest, ConstexprFunctionsCompile) {
  constexpr auto p = detail::periodic_index(-1, 10);
  constexpr auto r = detail::reflective_index(-1, 10);
  static_assert(p == 9, "periodic_index constexpr failed");
  static_assert(r == 1, "reflective_index constexpr failed");
  (void)p;
  (void)r;
}

}  // namespace pyinterp::fill
