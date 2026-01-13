// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/fill/matrix.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace pyinterp::fill {

TEST(FillLineTest, AllUndefinedNoChange) {
  Vector<double> x(4);
  x << 1.0, 2.0, 3.0, 4.0;
  Vector<bool> mask(4);
  mask.setConstant(true);

  detail::fill_line<double>(x, mask);

  EXPECT_TRUE(mask.all());
  EXPECT_DOUBLE_EQ(x[0], 1.0);
  EXPECT_DOUBLE_EQ(x[1], 2.0);
  EXPECT_DOUBLE_EQ(x[2], 3.0);
  EXPECT_DOUBLE_EQ(x[3], 4.0);
}

TEST(FillLineTest, SingleValidFillsConstant) {
  Vector<double> x(5);
  x.setConstant(-1.0);
  Vector<bool> mask(5);
  mask.setConstant(true);

  x[2] = 10.0;
  mask[2] = false;

  detail::fill_line<double>(x, mask);

  EXPECT_FALSE(mask.any());
  for (double i : x) {
    EXPECT_DOUBLE_EQ(i, 10.0);
  }
}

TEST(FillLineTest, InterpolateAndExtrapolate) {
  Vector<double> x(6);
  x.setZero();
  Vector<bool> mask(6);
  mask.setConstant(true);

  x[1] = 10.0;
  x[4] = 22.0;
  mask[1] = false;
  mask[4] = false;

  detail::fill_line<double>(x, mask);

  EXPECT_FALSE(mask.any());
  EXPECT_DOUBLE_EQ(x[0], 6.0);
  EXPECT_DOUBLE_EQ(x[1], 10.0);
  EXPECT_DOUBLE_EQ(x[2], 14.0);
  EXPECT_DOUBLE_EQ(x[3], 18.0);
  EXPECT_DOUBLE_EQ(x[4], 22.0);
  EXPECT_DOUBLE_EQ(x[5], 26.0);
}

TEST(FillMatrixTest, FillsRowsThenColumnsWithNaNMask) {
  Matrix<double> x(3, 3);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  x << 1.0, nan, 3.0,  // row 0
      nan, nan, nan,   // row 1
      7.0, 8.0, nan;   // row 2

  matrix<double>(x, nan);

  Matrix<double> expected(3, 3);
  expected << 1.0, 2.0, 3.0,  // row 0 after row fill
      4.0, 5.0, 6.0,          // filled by column pass
      7.0, 8.0, 9.0;

  for (int r = 0; r < x.rows(); ++r) {
    for (int c = 0; c < x.cols(); ++c) {
      EXPECT_DOUBLE_EQ(x(r, c), expected(r, c)) << "r=" << r << " c=" << c;
    }
  }
}

TEST(FillMatrixTest, FillsWithExplicitFillValue) {
  Matrix<double> x(3, 3);
  const double fill_value = -1.0;

  x << 0.0, fill_value, 2.0,               // row 0
      fill_value, fill_value, fill_value,  // row 1
      2.0, fill_value, 4.0;                // row 2

  matrix<double>(x, fill_value);

  Matrix<double> expected(3, 3);
  expected << 0.0, 1.0, 2.0,  // row 0 after row fill
      1.0, 2.0, 3.0,          // filled by column pass
      2.0, 3.0, 4.0;

  for (int r = 0; r < x.rows(); ++r) {
    for (int c = 0; c < x.cols(); ++c) {
      EXPECT_DOUBLE_EQ(x(r, c), expected(r, c)) << "r=" << r << " c=" << c;
    }
  }
}

TEST(FillVectorTest, FillsGaps) {
  Vector<double> v(5);
  v << 1.0, -1.0, 3.0, -1.0, 5.0;

  vector<double>(v, -1.0);

  Vector<double> expected(5);
  expected << 1.0, 2.0, 3.0, 4.0, 5.0;

  for (int i = 0; i < v.size(); ++i) {
    EXPECT_DOUBLE_EQ(v[i], expected[i]);
  }
}

TEST(FillVectorTest, AllMissingStaysUnchanged) {
  Vector<double> v(3);
  v.setConstant(-1.0);

  vector<double>(v, -1.0);

  for (double i : v) {
    EXPECT_DOUBLE_EQ(i, -1.0);
  }
}

TEST(FillVectorTest, SingleValidReplicates) {
  Vector<double> v(4);
  v.setConstant(-1.0);
  v[1] = 10.0;

  vector<double>(v, -1.0);

  for (double i : v) {
    EXPECT_DOUBLE_EQ(i, 10.0);
  }
}

}  // namespace pyinterp::fill
