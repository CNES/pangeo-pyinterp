#include <gtest/gtest.h>

#include "pyinterp/detail/math/dct2d.hpp"

namespace math = pyinterp::detail::math;

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

inline auto dct_data() -> pyinterp::RowMajorMatrix<double> {
  pyinterp::RowMajorMatrix<double> data(8, 4);
  data << 0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548, 0.64589411,
      0.43758721, 0.891773, 0.96366276, 0.38344152, 0.79172504, 0.52889492,
      0.56804456, 0.92559664, 0.07103606, 0.0871293, 0.0202184, 0.83261985,
      0.77815675, 0.87001215, 0.97861834, 0.79915856, 0.46147936, 0.78052918,
      0.11827443, 0.63992102, 0.14335329, 0.94466892, 0.52184832, 0.41466194,
      0.26455561, 0.77423369;
  return data;
}

TEST(DCT2D, RoundtripDouble) {
  RecordProperty("description",
                 "Test 2D DCT forward/inverse roundtrip for double");
  RecordProperty("type", "Nominal");

  constexpr int64_t rows = 8;
  constexpr int64_t cols = 4;

  auto data = dct_data();
  auto expected = data;  // The expected result is the original data

  // Create the DCT plan
  math::DCT2D<double> dct(rows, cols);

  // Perform roundtrip
  dct.forward(data);
  dct.inverse(data);  // Normalization is handled inside

  // Compare
  compare_matrices(data, expected, 1e-12);
}

TEST(DCT2D, RoundtripFloat) {
  RecordProperty("description",
                 "Test 2D DCT forward/inverse roundtrip for float");
  RecordProperty("type", "Nominal");

  constexpr int64_t rows = 8;
  constexpr int64_t cols = 4;

  pyinterp::RowMajorMatrix<float> data = dct_data().cast<float>();
  auto expected = data;  // The expected result is the original data

  // Create the DCT plan
  math::DCT2D<float> dct(rows, cols);

  // Perform roundtrip
  dct.forward(data);
  dct.inverse(data);  // Normalization is handled inside

  // Compare
  compare_matrices(data, expected, 1e-6f);
}
