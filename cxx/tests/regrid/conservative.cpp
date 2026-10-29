// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/regrid/conservative.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace pyinterp::regrid {

namespace {

/// Bounds of n contiguous cells of the given step, starting at origin
auto make_bounds(const double origin, const double step, const int64_t n)
    -> RowMajorMatrix<double> {
  RowMajorMatrix<double> result(n, 2);
  for (int64_t ix = 0; ix < n; ++ix) {
    result(ix, 0) = origin + static_cast<double>(ix) * step;
    result(ix, 1) = origin + static_cast<double>(ix + 1) * step;
  }
  return result;
}

}  // namespace

TEST(Conservative, OverlapMatrix) {
  const auto source = make_bounds(0, 1, 10);
  const auto target = make_bounds(-0.5, 2.5, 5);
  const auto matrix = overlap_matrix(source, target);
  ASSERT_EQ(matrix.rows(), 5);
  // [-0.5, 2]: half covered
  const auto sums = matrix.row_sums();
  EXPECT_DOUBLE_EQ(sums[0], 2.0 / 2.5);
  EXPECT_DOUBLE_EQ(sums[1], 1.0);
  EXPECT_DOUBLE_EQ(sums[3], 1.0);
  // [9.5, 12]: 0.5 covered
  EXPECT_DOUBLE_EQ(sums[4], 0.5 / 2.5);
}

TEST(Conservative, Periodic) {
  const auto source = make_bounds(0, 10, 36);
  RowMajorMatrix<double> target(2, 2);
  target << -5, 5, 715, 725;
  const auto matrix = overlap_matrix(source, target, 360);
  for (int64_t row = 0; row < 2; ++row) {
    ASSERT_EQ(matrix.indptr[row + 1] - matrix.indptr[row], 2);
    EXPECT_EQ(matrix.indices[matrix.indptr[row]], 35);
    EXPECT_EQ(matrix.indices[matrix.indptr[row] + 1], 0);
    EXPECT_DOUBLE_EQ(matrix.weights[matrix.indptr[row]], 0.5);
  }
  target.resize(1, 2);
  target << 0, 370;
  EXPECT_THROW(static_cast<void>(overlap_matrix(source, target, 360)),
               std::invalid_argument);
}

TEST(Conservative, Overlapping) {
  RowMajorMatrix<double> source(2, 2);
  source << 0, 2, 1, 3;
  EXPECT_THROW(static_cast<void>(overlap_matrix(source, source)),
               std::invalid_argument);
}

TEST(Conservative, AuthalicLatitude) {
  const auto bounds = make_bounds(-90, 30, 6);
  const auto sphere = latitude_measure(bounds, std::nullopt);
  const auto wgs84 = latitude_measure(bounds, geometry::geographic::Spheroid{});
  EXPECT_DOUBLE_EQ(sphere(0, 0), -1);
  EXPECT_DOUBLE_EQ(wgs84(0, 0), -1);
  EXPECT_DOUBLE_EQ(wgs84(5, 1), 1);
  EXPECT_NEAR(wgs84(2, 1), 0, 1e-16);
  // The authalic latitude is smaller than the geodetic latitude
  EXPECT_LT(wgs84(4, 0), sphere(4, 0));
}

TEST(Conservative, BlockMean) {
  // Coarsen a 4 x 6 grid by 2 x 3, with an undefined value
  const auto x = overlap_matrix(make_bounds(0, 1, 4), make_bounds(0, 2, 2));
  const auto y = overlap_matrix(make_bounds(0, 1, 6), make_bounds(0, 3, 2));
  RowMajorMatrix<double> data(8, 6);
  for (int64_t ix = 0; ix < data.size(); ++ix) {
    data.data()[ix] = static_cast<double>(ix);
  }
  // Second field: one undefined value in the first block
  data(4, 0) = std::numeric_limits<double>::quiet_NaN();
  const auto result = conservative<double>(x, y, data, 4, 0, true, 1);
  ASSERT_EQ(result.rows(), 4);
  ASSERT_EQ(result.cols(), 2);
  EXPECT_DOUBLE_EQ(result(0, 0), (0 + 1 + 2 + 6 + 7 + 8) / 6.0);
  EXPECT_DOUBLE_EQ(result(1, 1), (15 + 16 + 17 + 21 + 22 + 23) / 6.0);
  EXPECT_DOUBLE_EQ(result(2, 0), (25 + 26 + 30 + 31 + 32) / 5.0);
  const auto threshold = conservative<double>(x, y, data, 4, 0.9, true, 0);
  EXPECT_TRUE(std::isnan(threshold(2, 0)));
  EXPECT_DOUBLE_EQ(threshold(0, 0), result(0, 0));
  const auto zero = conservative<double>(x, y, data, 4, 0, false, 2);
  EXPECT_DOUBLE_EQ(zero(2, 0), (25 + 26 + 30 + 31 + 32) / 6.0);
  EXPECT_THROW(
      static_cast<void>(conservative<double>(x, y, data, 3, 0, true, 1)),
      std::invalid_argument);
}

}  // namespace pyinterp::regrid
