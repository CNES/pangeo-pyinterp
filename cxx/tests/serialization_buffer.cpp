// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/serialization_buffer.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <limits>
#include <memory>
#include <numbers>
#include <vector>

namespace pyinterp::serialization {

// ============================================================================
// Writer Tests
// ============================================================================

class WriterTest : public ::testing::Test {
 protected:
  Writer writer_;
};

TEST_F(WriterTest, DefaultConstructorCreatesEmptyBuffer) {
  EXPECT_EQ(writer_.size(), 0);
  EXPECT_EQ(writer_.data(), nullptr);
}

TEST_F(WriterTest, WriteTriviallyCopiableInt) {
  int value = 42;
  writer_.write(value);
  EXPECT_EQ(writer_.size(), sizeof(int));
}

TEST_F(WriterTest, WriteTriviallyCopiableDouble) {
  double value = std::numbers::pi;
  writer_.write(value);
  EXPECT_EQ(writer_.size(), sizeof(double));
}

TEST_F(WriterTest, WriteTriviallyCopiableChar) {
  char value = 'A';
  writer_.write(value);
  EXPECT_EQ(writer_.size(), sizeof(char));
}

TEST_F(WriterTest, WriteMultipleTriviallyCopiableValues) {
  int int_val = 42;
  double double_val = 3.14;
  char char_val = 'X';

  writer_.write(int_val);
  writer_.write(double_val);
  writer_.write(char_val);

  EXPECT_EQ(writer_.size(), sizeof(int) + sizeof(double) + sizeof(char));
}

TEST_F(WriterTest, WriteEmptyVector) {
  std::vector<int> vec;
  writer_.write(vec);
  // Should write size (size_t = 8 bytes on 64-bit)
  EXPECT_EQ(writer_.size(), sizeof(size_t));
}

TEST_F(WriterTest, WriteVectorOfTriviallyCopyableInts) {
  std::vector<int> vec = {1, 2, 3, 4, 5};
  writer_.write(vec);
  // Size (size_t) + 5 ints
  EXPECT_EQ(writer_.size(), sizeof(size_t) + 5 * sizeof(int));
}

TEST_F(WriterTest, WriteVectorOfTriviallyCopyableDoubles) {
  std::vector<double> vec = {1.1, 2.2, 3.3};
  writer_.write(vec);
  // Size (size_t) + 3 doubles
  EXPECT_EQ(writer_.size(), sizeof(size_t) + 3 * sizeof(double));
}

TEST_F(WriterTest, WriteEmptyString) {
  std::string str = "";
  writer_.write(str);
  // Should write size only (empty string)
  EXPECT_EQ(writer_.size(), sizeof(size_t));
}

TEST_F(WriterTest, WriteSimpleString) {
  std::string str = "Hello";
  writer_.write(str);
  // Size (size_t) + 5 chars
  EXPECT_EQ(writer_.size(), sizeof(size_t) + 5);
}

TEST_F(WriterTest, WriteLongString) {
  std::string str = "This is a longer test string with spaces and punctuation!";
  writer_.write(str);
  // Size (size_t) + string length
  EXPECT_EQ(writer_.size(), sizeof(size_t) + str.size());
}

TEST_F(WriterTest, WriteEigenMatrixDynamic) {
  Eigen::MatrixXd mat(2, 3);
  mat << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  writer_.write(mat);
  // Size for rows + cols + storage order + 6 doubles
  EXPECT_EQ(writer_.size(),
            2 * sizeof(size_t) + sizeof(char) + 6 * sizeof(double));
}

TEST_F(WriterTest, WriteEigenMatrixDynamicRowMajor) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(2,
                                                                             3);
  mat << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  writer_.write(mat);
  // Size for rows + cols + storage order + 6 doubles
  EXPECT_EQ(writer_.size(),
            2 * sizeof(size_t) + sizeof(char) + 6 * sizeof(double));
}

TEST_F(WriterTest, WriteEigenVector) {
  Eigen::VectorXd vec(5);
  vec << 1.0, 2.0, 3.0, 4.0, 5.0;

  writer_.write(vec);
  // Size for rows + cols + storage order + 5 doubles
  EXPECT_EQ(writer_.size(),
            2 * sizeof(size_t) + sizeof(char) + 5 * sizeof(double));
}

TEST_F(WriterTest, WriteEigenMatrixFixed) {
  Eigen::Matrix2d mat;
  mat << 1.0, 2.0, 3.0, 4.0;

  writer_.write(mat);
  // Size for rows + cols + storage order + 4 doubles
  EXPECT_EQ(writer_.size(),
            2 * sizeof(size_t) + 4 * sizeof(double) + sizeof(char));
}

TEST_F(WriterTest, GetDataReturnsValidPointer) {
  writer_.write(42);
  EXPECT_NE(writer_.data(), nullptr);
  EXPECT_EQ(writer_.size(), sizeof(int));
}

TEST_F(WriterTest, GetSpanReturnsValidSpan) {
  writer_.write(42);
  auto span = writer_.span();
  EXPECT_FALSE(span.empty());
  EXPECT_EQ(span.size(), sizeof(int));
}

TEST_F(WriterTest, ReleaseTransfersOwnership) {
  writer_.write(42);
  auto size_before = writer_.size();
  auto buffer = std::move(writer_).release();

  EXPECT_EQ(buffer.size(), size_before);
  EXPECT_FALSE(buffer.empty());
}

TEST_F(WriterTest, ClearEmptiesBuffer) {
  writer_.write(42);
  EXPECT_GT(writer_.size(), 0);

  writer_.clear();
  EXPECT_EQ(writer_.size(), 0);
}

TEST_F(WriterTest, ClearMultipleTimes) {
  writer_.write(42);
  writer_.clear();
  writer_.write(3.14);
  writer_.clear();
  EXPECT_EQ(writer_.size(), 0);
}

TEST_F(WriterTest, WriteComplexDataStructure) {
  writer_.write(42);
  writer_.write(3.14);
  writer_.write(std::string("test"));
  std::vector<int> vec = {1, 2, 3};
  writer_.write(vec);

  size_t expected_size = sizeof(int) + sizeof(double) + sizeof(size_t) + 4 +
                         sizeof(size_t) + 3 * sizeof(int);
  EXPECT_EQ(writer_.size(), expected_size);
}

// ============================================================================
// Reader Tests
// ============================================================================

class ReaderTest : public ::testing::Test {
 protected:
  Writer writer_;
  Reader reader_;
};

TEST_F(ReaderTest, DefaultConstructorCreatesEmptyReader) {
  EXPECT_EQ(reader_.size(), 0);
  EXPECT_EQ(reader_.tell(), 0);
  EXPECT_FALSE(reader_.has_owner());
}

TEST_F(ReaderTest, ConstructFromWriter) {
  writer_.write(42);
  auto size_before = writer_.size();

  Reader reader(std::move(writer_));
  EXPECT_EQ(reader.size(), size_before);
  EXPECT_TRUE(reader.has_owner());
  EXPECT_EQ(reader.tell(), 0);
}

TEST_F(ReaderTest, ConstructFromVector) {
  std::vector<std::byte> buffer(10);
  size_t original_size = buffer.size();

  Reader reader(std::move(buffer));
  EXPECT_EQ(reader.size(), original_size);
  EXPECT_TRUE(reader.has_owner());
}

TEST_F(ReaderTest, ConstructFromRawPointer) {
  std::vector<int> data = {1, 2, 3, 4, 5};
  auto bytes = std::as_bytes(std::span{data});

  Reader reader(bytes.data(), bytes.size());
  EXPECT_EQ(reader.size(), bytes.size());
  EXPECT_FALSE(reader.has_owner());
}

TEST_F(ReaderTest, ConstructWithCustomOwner) {
  auto owner = std::make_shared<std::vector<std::byte>>(10);
  const auto* data = owner->data();

  Reader reader(data, owner->size(), owner);
  EXPECT_EQ(reader.size(), 10);
  EXPECT_TRUE(reader.has_owner());
}

TEST_F(ReaderTest, GetData) {
  writer_.write(42);
  Reader reader(std::move(writer_));

  EXPECT_NE(reader.data(), nullptr);
}

TEST_F(ReaderTest, GetSize) {
  writer_.write(42);
  writer_.write(3.14);
  Reader reader(std::move(writer_));

  size_t expected = sizeof(int) + sizeof(double);
  EXPECT_EQ(reader.size(), expected);
}

TEST_F(ReaderTest, TellInitiallyZero) {
  writer_.write(42);
  Reader reader(std::move(writer_));

  EXPECT_EQ(reader.tell(), 0);
}

TEST_F(ReaderTest, SeekToValidPosition) {
  writer_.write(42);
  writer_.write(3.14);
  Reader reader(std::move(writer_));

  reader.seek(sizeof(int));
  EXPECT_EQ(reader.tell(), sizeof(int));
}

TEST_F(ReaderTest, SeekToZero) {
  writer_.write(42);
  writer_.write(3.14);
  Reader reader(std::move(writer_));

  reader.seek(sizeof(int));
  reader.seek(0);
  EXPECT_EQ(reader.tell(), 0);
}

TEST_F(ReaderTest, SeekToEndOfBuffer) {
  writer_.write(42);
  writer_.write(3.14);
  Reader reader(std::move(writer_));
  size_t buffer_size = reader.size();

  reader.seek(buffer_size);
  EXPECT_EQ(reader.tell(), buffer_size);
}

TEST_F(ReaderTest, SeekBeyondBufferThrows) {
  writer_.write(42);
  Reader reader(std::move(writer_));

  EXPECT_THROW(reader.seek(reader.size() + 1), std::out_of_range);
}

TEST_F(ReaderTest, ResetReadPosition) {
  writer_.write(42);
  writer_.write(3.14);
  Reader reader(std::move(writer_));

  reader.seek(sizeof(int));
  EXPECT_EQ(reader.tell(), sizeof(int));

  reader.reset();
  EXPECT_EQ(reader.tell(), 0);
}

TEST_F(ReaderTest, ReadInt) {
  int original = 42;
  writer_.write(original);

  Reader reader(std::move(writer_));
  int read_value = reader.read<int>();

  EXPECT_EQ(read_value, original);
  EXPECT_EQ(reader.tell(), sizeof(int));
}

TEST_F(ReaderTest, ReadDouble) {
  double original = std::numbers::pi;
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_value = reader.read<double>();

  EXPECT_DOUBLE_EQ(read_value, original);
}

TEST_F(ReaderTest, ReadChar) {
  char original = 'Z';
  writer_.write(original);

  Reader reader(std::move(writer_));
  char read_value = reader.read<char>();

  EXPECT_EQ(read_value, original);
}

TEST_F(ReaderTest, ReadMultipleValues) {
  int int_val = 42;
  double double_val = 3.14;
  char char_val = 'X';

  writer_.write(int_val);
  writer_.write(double_val);
  writer_.write(char_val);

  Reader reader(std::move(writer_));

  int read_int = reader.read<int>();
  auto read_double = reader.read<double>();
  char read_char = reader.read<char>();

  EXPECT_EQ(read_int, int_val);
  EXPECT_DOUBLE_EQ(read_double, double_val);
  EXPECT_EQ(read_char, char_val);
}

TEST_F(ReaderTest, ReadPastBufferThrows) {
  writer_.write(42);

  Reader reader(std::move(writer_));
  auto read_int = reader.read<int>();  // Read the only int
  EXPECT_EQ(read_int, 42);

  EXPECT_THROW(static_cast<void>(reader.read<int>()), std::out_of_range);
}

TEST_F(ReaderTest, ReadEmptyVector) {
  std::vector<int> original;
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_vec = reader.read_vector<int>();

  EXPECT_EQ(read_vec.size(), 0);
}

TEST_F(ReaderTest, ReadVectorOfInts) {
  std::vector<int> original = {1, 2, 3, 4, 5};
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_vec = reader.read_vector<int>();

  EXPECT_EQ(read_vec, original);
}

TEST_F(ReaderTest, ReadVectorOfDoubles) {
  std::vector<double> original = {1.1, 2.2, 3.3, 4.4};
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_vec = reader.read_vector<double>();

  EXPECT_EQ(read_vec.size(), original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_DOUBLE_EQ(read_vec[i], original[i]);
  }
}

TEST_F(ReaderTest, ReadVectorPastBufferThrows) {
  std::vector<int> original = {1, 2, 3};
  writer_.write(original);

  Reader reader(std::move(writer_));
  static_cast<void>(reader.read_vector<int>());  // Read the vector

  EXPECT_THROW(static_cast<void>(reader.read_vector<int>()), std::out_of_range);
}

TEST_F(ReaderTest, ReadEmptyString) {
  std::string original = "";
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_str = reader.read_string();

  EXPECT_EQ(read_str, original);
}

TEST_F(ReaderTest, ReadSimpleString) {
  std::string original = "Hello";
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_str = reader.read_string();

  EXPECT_EQ(read_str, original);
}

TEST_F(ReaderTest, ReadLongString) {
  std::string original =
      "The quick brown fox jumps over the lazy dog. This is a test!";
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_str = reader.read_string();

  EXPECT_EQ(read_str, original);
}

TEST_F(ReaderTest, ReadStringWithSpecialCharacters) {
  std::string original = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`";
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_str = reader.read_string();

  EXPECT_EQ(read_str, original);
}

TEST_F(ReaderTest, ReadEigenMatrixDynamic) {
  Eigen::MatrixXd original(2, 3);
  original << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_mat = reader.read_eigen<double>();

  EXPECT_EQ(read_mat.rows(), 2);
  EXPECT_EQ(read_mat.cols(), 3);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(read_mat(i, j), original(i, j));
    }
  }
}

TEST_F(ReaderTest, ReadEigenMatrixRowMajor) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      original(2, 3);
  original << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_mat = reader.read_eigen<double>();

  EXPECT_EQ(read_mat.rows(), 2);
  EXPECT_EQ(read_mat.cols(), 3);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(read_mat(i, j), original(i, j));
    }
  }
}

TEST_F(ReaderTest, ReadEigenVector) {
  Eigen::VectorXd original(5);
  original << 1.0, 2.0, 3.0, 4.0, 5.0;

  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_vec = reader.read_eigen<double>();

  EXPECT_EQ(read_vec.rows(), 5);
  EXPECT_EQ(read_vec.cols(), 1);
  for (int i = 0; i < 5; ++i) {
    EXPECT_DOUBLE_EQ(read_vec(i), original(i));
  }
}

TEST_F(ReaderTest, ReadEigenMatrixFixed) {
  Eigen::Matrix2d original;
  original << 1.0, 2.0, 3.0, 4.0;

  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_mat = reader.read_eigen<double>();

  EXPECT_EQ(read_mat.rows(), 2);
  EXPECT_EQ(read_mat.cols(), 2);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_DOUBLE_EQ(read_mat(i, j), original(i, j));
    }
  }
}

TEST_F(ReaderTest, ReadEigenMatrixLarge) {
  Eigen::MatrixXd original(10, 20);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      original(i, j) = i * 20.0 + j + 0.5;
    }
  }

  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_mat = reader.read_eigen<double>();

  EXPECT_EQ(read_mat.rows(), 10);
  EXPECT_EQ(read_mat.cols(), 20);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      EXPECT_DOUBLE_EQ(read_mat(i, j), original(i, j));
    }
  }
}

// ============================================================================
// Round-trip Tests
// ============================================================================

class RoundTripTest : public ::testing::Test {
 protected:
  Writer writer_;
};

TEST_F(RoundTripTest, RoundTripInt) {
  int original = 42;
  writer_.write(original);

  Reader reader(std::move(writer_));
  int read_value = reader.read<int>();

  EXPECT_EQ(read_value, original);
}

TEST_F(RoundTripTest, RoundTripMultiplePrimitives) {
  int int_val = 100;
  double double_val = std::numbers::e;
  char char_val = 'Q';

  writer_.write(int_val);
  writer_.write(double_val);
  writer_.write(char_val);

  Reader reader(std::move(writer_));

  int read_int = reader.read<int>();
  auto read_double = reader.read<double>();
  char read_char = reader.read<char>();

  EXPECT_EQ(read_int, int_val);
  EXPECT_DOUBLE_EQ(read_double, double_val);
  EXPECT_EQ(read_char, char_val);
}

TEST_F(RoundTripTest, RoundTripVector) {
  std::vector<int> original = {10, 20, 30, 40, 50};
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_vec = reader.read_vector<int>();

  EXPECT_EQ(read_vec, original);
}

TEST_F(RoundTripTest, RoundTripString) {
  std::string original = "Round trip test string";
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_str = reader.read_string();

  EXPECT_EQ(read_str, original);
}

TEST_F(RoundTripTest, RoundTripEigenMatrix) {
  Eigen::MatrixXd original(3, 4);
  original << 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12;

  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_mat = reader.read_eigen<double>();

  EXPECT_EQ(read_mat.rows(), original.rows());
  EXPECT_EQ(read_mat.cols(), original.cols());
  for (int i = 0; i < original.rows(); ++i) {
    for (int j = 0; j < original.cols(); ++j) {
      EXPECT_DOUBLE_EQ(read_mat(i, j), original(i, j));
    }
  }
}

TEST_F(RoundTripTest, RoundTripComplexMixedData) {
  // Write various data types
  int int_val = 255;
  double double_val = std::numbers::sqrt2;
  std::string str = "Complex test";
  std::vector<int> vec = {1, 1, 2, 3, 5, 8, 13};
  Eigen::MatrixXd mat(2, 2);
  mat << 1.0, 2.0, 3.0, 4.0;

  writer_.write(int_val);
  writer_.write(double_val);
  writer_.write(str);
  writer_.write(vec);
  writer_.write(mat);

  // Read back
  Reader reader(std::move(writer_));

  int read_int = reader.read<int>();
  auto read_double = reader.read<double>();
  std::string read_str = reader.read_string();
  auto read_vec = reader.read_vector<int>();
  auto read_mat = reader.read_eigen<double>();

  // Verify
  EXPECT_EQ(read_int, int_val);
  EXPECT_DOUBLE_EQ(read_double, double_val);
  EXPECT_EQ(read_str, str);
  EXPECT_EQ(read_vec, vec);
  EXPECT_EQ(read_mat.rows(), mat.rows());
  EXPECT_EQ(read_mat.cols(), mat.cols());
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      EXPECT_DOUBLE_EQ(read_mat(i, j), mat(i, j));
    }
  }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

class EdgeCaseTest : public ::testing::Test {
 protected:
  Writer writer_;
};

TEST_F(EdgeCaseTest, WriteZeroValue) {
  int value = 0;
  writer_.write(value);

  Reader reader(std::move(writer_));
  int read_value = reader.read<int>();

  EXPECT_EQ(read_value, 0);
}

TEST_F(EdgeCaseTest, WriteNegativeValue) {
  int value = -12345;
  writer_.write(value);

  Reader reader(std::move(writer_));
  int read_value = reader.read<int>();

  EXPECT_EQ(read_value, value);
}

TEST_F(EdgeCaseTest, WriteMaxIntValue) {
  int value = std::numeric_limits<int>::max();
  writer_.write(value);

  Reader reader(std::move(writer_));
  int read_value = reader.read<int>();

  EXPECT_EQ(read_value, value);
}

TEST_F(EdgeCaseTest, WriteMinIntValue) {
  int value = std::numeric_limits<int>::min();
  writer_.write(value);

  Reader reader(std::move(writer_));
  int read_value = reader.read<int>();

  EXPECT_EQ(read_value, value);
}

TEST_F(EdgeCaseTest, WriteVerySmallDouble) {
  double value = std::numeric_limits<double>::min();
  writer_.write(value);

  Reader reader(std::move(writer_));
  auto read_value = reader.read<double>();

  EXPECT_DOUBLE_EQ(read_value, value);
}

TEST_F(EdgeCaseTest, WriteLargeVectorOfInts) {
  std::vector<int> original(10000);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<int>(i);
  }

  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_vec = reader.read_vector<int>();

  EXPECT_EQ(read_vec.size(), original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_EQ(read_vec[i], original[i]);
  }
}

TEST_F(EdgeCaseTest, SingleCharacterString) {
  std::string original = "X";
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_str = reader.read_string();

  EXPECT_EQ(read_str, original);
}

TEST_F(EdgeCaseTest, StringWithNullCharacters) {
  // Note: std::string can contain null characters
  std::string original("Hello\0World", 11);
  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_str = reader.read_string();

  EXPECT_EQ(read_str, original);
  EXPECT_EQ(read_str.size(), 11);
}

TEST_F(EdgeCaseTest, WriteSingleElementMatrix) {
  Eigen::MatrixXd original(1, 1);
  original << 42.0;

  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_mat = reader.read_eigen<double>();

  EXPECT_EQ(read_mat.rows(), 1);
  EXPECT_EQ(read_mat.cols(), 1);
  EXPECT_DOUBLE_EQ(read_mat(0, 0), 42.0);
}

TEST_F(EdgeCaseTest, WriteRowVectorMatrix) {
  Eigen::MatrixXd original(1, 10);
  for (int i = 0; i < 10; ++i) {
    original(0, i) = i * 1.5;
  }

  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_mat = reader.read_eigen<double>();

  EXPECT_EQ(read_mat.rows(), 1);
  EXPECT_EQ(read_mat.cols(), 10);
  for (int i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(read_mat(0, i), original(0, i));
  }
}

TEST_F(EdgeCaseTest, WriteColumnVectorMatrix) {
  Eigen::MatrixXd original(10, 1);
  for (int i = 0; i < 10; ++i) {
    original(i, 0) = i * 2.5;
  }

  writer_.write(original);

  Reader reader(std::move(writer_));
  auto read_mat = reader.read_eigen<double>();

  EXPECT_EQ(read_mat.rows(), 10);
  EXPECT_EQ(read_mat.cols(), 1);
  for (int i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(read_mat(i, 0), original(i, 0));
  }
}

// ============================================================================
// Copy/Move Semantics Tests
// ============================================================================

class SemanticsTest : public ::testing::Test {};

TEST_F(SemanticsTest, ReaderMoveConstruction) {
  Writer writer;
  writer.write(42);

  Reader reader1(std::move(writer));
  EXPECT_TRUE(reader1.has_owner());

  Reader reader2 = std::move(reader1);
  EXPECT_TRUE(reader2.has_owner());
}

TEST_F(SemanticsTest, ReaderMoveAssignment) {
  Writer writer1;
  writer1.write(42);

  Reader reader1(std::move(writer1));

  Writer writer2;
  writer2.write(3.14);

  Reader reader2(std::move(writer2));

  reader1 = std::move(reader2);
  EXPECT_TRUE(reader1.has_owner());
}

TEST_F(SemanticsTest, ReaderCopyConstructorDeleted) {
  Writer writer;
  writer.write(42);

  Reader reader1(std::move(writer));

  // This should not compile, but we test the concept
  // Reader reader2 = reader1;  // Should fail
  static_assert(!std::is_copy_constructible_v<Reader>,
                "Reader should not be copy constructible");
}

TEST_F(SemanticsTest, ReaderCopyAssignmentDeleted) {
  // This should not compile
  static_assert(!std::is_copy_assignable_v<Reader>,
                "Reader should not be copy assignable");
}

TEST_F(SemanticsTest, ReaderMoveConstructible) {
  static_assert(std::is_move_constructible_v<Reader>,
                "Reader should be move constructible");
}

TEST_F(SemanticsTest, ReaderMoveAssignable) {
  static_assert(std::is_move_assignable_v<Reader>,
                "Reader should be move assignable");
}

}  // namespace pyinterp::serialization
