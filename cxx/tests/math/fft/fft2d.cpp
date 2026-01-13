// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/fft2d.hpp"

#include <gtest/gtest.h>

namespace pyinterp::math {

// Function to compare two Eigen matrices with a tolerance
template <typename T>
void compare_matrices(const pyinterp::RowMajorMatrix<T>& result,
                      const pyinterp::RowMajorMatrix<T>& expected,
                      const T tolerance) {
  ASSERT_EQ(result.rows(), expected.rows());
  ASSERT_EQ(result.cols(), expected.cols());
  for (int64_t ix = 0; ix < result.rows(); ++ix) {
    for (int64_t iy = 0; iy < result.cols(); ++iy) {
      EXPECT_NEAR(result(ix, iy), expected(ix, iy), tolerance);
    }
  }
}

inline auto fft_data() -> RowMajorMatrix<double> {
  RowMajorMatrix<double> data(8, 4);
  data << 0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548, 0.64589411,
      0.43758721, 0.891773, 0.96366276, 0.38344152, 0.79172504, 0.52889492,
      0.56804456, 0.92559664, 0.07103606, 0.0871293, 0.0202184, 0.83261985,
      0.77815675, 0.87001215, 0.97861834, 0.79915856, 0.46147936, 0.78052918,
      0.11827443, 0.63992102, 0.14335329, 0.94466892, 0.52184832, 0.41466194,
      0.26455561, 0.77423369;
  return data;
}

TEST(FFT2D, RoundtripDouble) {
  RecordProperty("description",
                 "Test 2D FFT forward/inverse roundtrip for double");
  RecordProperty("type", "Nominal");

  constexpr int64_t rows = 8;
  constexpr int64_t cols = 4;

  auto data = fft_data();
  auto expected = data;  // The expected result is the original data

  // Create the FFT plan
  FFT2D<double> fft(rows, cols);
  RowMajorComplexMatrix<double> c_data(rows, fft.c_cols());

  // Perform roundtrip
  fft.forward(data, c_data);
  fft.inverse(c_data, data);  // Normalization is handled inside

  // Compare
  compare_matrices(data, expected, 1e-12);
}

TEST(FFT2D, RoundtripFloat) {
  RecordProperty("description",
                 "Test 2D FFT forward/inverse roundtrip for float");
  RecordProperty("type", "Nominal");

  constexpr int64_t rows = 8;
  constexpr int64_t cols = 4;

  RowMajorMatrix<float> data = fft_data().cast<float>();
  auto expected = data;  // The expected result is the original data

  // Create the DCT plan
  FFT2D<float> fft(rows, cols);
  RowMajorComplexMatrix<float> c_data(rows, fft.c_cols());

  // Perform roundtrip
  fft.forward(data, c_data);
  fft.inverse(c_data, data);  // Normalization is handled inside

  // Compare
  compare_matrices(data, expected, 1e-6f);
}

}  // namespace pyinterp::math
