// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/geohash/base32.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

namespace pyinterp::geohash {

// Test fixture for Base32 tests
class Base32Test : public ::testing::Test {
 protected:
  void SetUp() override { encoder_ = Base32{}; }

  Base32 encoder_;
};

// Test constructor initializes decode table correctly
TEST_F(Base32Test, ConstructorInitialization) {
  // Constructor should not throw
  EXPECT_NO_THROW(Base32{});

  // Verify encoder is usable
  std::array<char, 1> buffer{'0'};
  EXPECT_TRUE(encoder_.validate(buffer));
}

// Test validation with valid base32 characters
TEST_F(Base32Test, ValidateValidCharacters) {
  // Test all valid base32 characters
  const std::string_view valid_chars = "0123456789bcdefghjkmnpqrstuvwxyz";

  for (char ch : valid_chars) {
    std::array<char, 2> buffer{ch, '\0'};
    EXPECT_TRUE(encoder_.validate(buffer))
        << "Character '" << ch << "' should be valid";
  }
}

// Test validation with valid geohash strings
TEST_F(Base32Test, ValidateValidGeohashes) {
  // Common geohash examples
  const std::array<std::string_view, 5> valid_hashes = {
      "ezs42",        // Example geohash
      "u4pruydqqvj",  // High precision geohash
      "9q5",          // Short geohash
      "gbsuv",        // Another example
      "0"             // Single character
  };

  for (const auto& hash : valid_hashes) {
    std::vector<char> buffer(hash.begin(), hash.end());
    buffer.push_back('\0');
    EXPECT_TRUE(encoder_.validate(buffer))
        << "Geohash '" << hash << "' should be valid";
  }
}

// Test validation with invalid characters
TEST_F(Base32Test, ValidateInvalidCharacters) {
  // Characters not in base32: a, i, l, o
  const std::array<char, 4> invalid_chars = {'a', 'i', 'l', 'o'};

  for (char ch : invalid_chars) {
    std::array<char, 2> buffer{ch, '\0'};
    EXPECT_FALSE(encoder_.validate(buffer))
        << "Character '" << ch << "' should be invalid";
  }

  // Test uppercase letters (base32 uses lowercase)
  std::array<char, 2> uppercase{'A', '\0'};
  EXPECT_FALSE(encoder_.validate(uppercase))
      << "Uppercase letters should be invalid";
}

// Test validation with mixed valid/invalid characters
TEST_F(Base32Test, ValidateMixedCharacters) {
  // Valid start, invalid character in middle
  std::array<char, 4> buffer1{'e', 'z', 'a', '\0'};
  EXPECT_FALSE(encoder_.validate(buffer1))
      << "String with invalid 'a' should be invalid";

  // Valid characters with invalid at end
  std::array<char, 4> buffer2{'e', 'z', 'A', '\0'};
  EXPECT_FALSE(encoder_.validate(buffer2))
      << "String with uppercase 'A' should be invalid";
}

// Test validation with empty string
TEST_F(Base32Test, ValidateEmptyString) {
  std::array<char, 1> buffer{'\0'};
  EXPECT_TRUE(encoder_.validate(buffer)) << "Empty string should be valid";
}

// Test decode with valid single character
TEST_F(Base32Test, DecodeSingleCharacter) {
  std::array<char, 2> buffer{'0', '\0'};
  auto [hash, count] = encoder_.decode(buffer);

  EXPECT_EQ(count, 1u) << "Should decode 1 character";
  EXPECT_EQ(hash, 0u) << "Character '0' should decode to 0";

  buffer[0] = '1';
  std::tie(hash, count) = encoder_.decode(buffer);
  EXPECT_EQ(hash, 1u) << "Character '1' should decode to 1";

  buffer[0] = 'z';
  std::tie(hash, count) = encoder_.decode(buffer);
  EXPECT_EQ(hash, 31u) << "Character 'z' should decode to 31";
}

// Test decode with multi-character geohash
TEST_F(Base32Test, DecodeMultipleCharacters) {
  // "ezs42" is a common example geohash
  const std::string geohash = "ezs42";
  std::vector<char> buffer(geohash.begin(), geohash.end());
  buffer.push_back('\0');

  auto [hash, count] = encoder_.decode(buffer);

  EXPECT_EQ(count, 5u) << "Should decode 5 characters";
  EXPECT_GT(hash, 0u) << "Hash should be non-zero";
}

// Test decode with maximum precision
TEST_F(Base32Test, DecodeMaxPrecision) {
  // 12 characters (60 bits, fits in 64-bit word)
  const std::string geohash = "u4pruydqqvj0";
  std::vector<char> buffer(geohash.begin(), geohash.end());
  buffer.push_back('\0');

  auto [hash, count] = encoder_.decode(buffer);

  EXPECT_EQ(count, 12u) << "Should decode all 12 characters";
  EXPECT_GT(hash, 0u) << "Hash should be non-zero";
}

// Test decode with empty string
TEST_F(Base32Test, DecodeEmptyString) {
  std::array<char, 1> buffer{'\0'};
  auto [hash, count] = encoder_.decode(buffer);

  EXPECT_EQ(count, 0u) << "Should decode 0 characters";
  EXPECT_EQ(hash, 0u) << "Hash should be 0 for empty string";
}

// Test decode throws on invalid character
TEST_F(Base32Test, DecodeInvalidCharacterThrows) {
  // String with invalid character 'a'
  std::array<char, 4> buffer{'e', 'z', 'a', '\0'};

  EXPECT_THROW(
      {
        try {
          static_cast<void>(encoder_.decode(buffer));
        } catch (const std::invalid_argument& e) {
          // Verify exception message contains useful info
          EXPECT_NE(std::string(e.what()).find("Invalid character"),
                    std::string::npos);
          throw;
        }
      },
      std::invalid_argument)
      << "Should throw invalid_argument for invalid character";
}

// Test decode throws on uppercase character
TEST_F(Base32Test, DecodeUppercaseThrows) {
  std::array<char, 2> buffer{'Z', '\0'};

  EXPECT_THROW(static_cast<void>(encoder_.decode(buffer)),
               std::invalid_argument)
      << "Should throw for uppercase character";
}

// Test encode with zero hash
TEST_F(Base32Test, EncodeZeroHash) {
  std::array<char, 5> buffer{};
  Base32::encode(0u, buffer);

  EXPECT_EQ(buffer[0], '0');
  EXPECT_EQ(buffer[1], '0');
  EXPECT_EQ(buffer[2], '0');
  EXPECT_EQ(buffer[3], '0');
  EXPECT_EQ(buffer[4], '0');
}

// Test encode with small hash value
TEST_F(Base32Test, EncodeSmallValue) {
  // Encode value 31 (0b11111) into single character
  std::array<char, 1> buffer{};
  Base32::encode(31u, buffer);

  EXPECT_EQ(buffer[0], 'z') << "31 should encode to 'z'";

  // Encode value 1 into single character
  Base32::encode(1u, buffer);
  EXPECT_EQ(buffer[0], '1') << "1 should encode to '1'";
}

// Test encode with multi-character output
TEST_F(Base32Test, EncodeMultipleCharacters) {
  std::array<char, 5> buffer{};

  // Encode a known hash value
  std::uint64_t hash = (13u << 20) | (30u << 15) | (25u << 10) | (4u << 5) |
                       2u;  // "ezs42" pattern
  Base32::encode(hash, buffer);

  // All characters should be valid base32
  for (char ch : buffer) {
    std::array<char, 2> test_buffer{ch, '\0'};
    EXPECT_TRUE(encoder_.validate(test_buffer))
        << "Encoded character '" << ch << "' should be valid";
  }
}

// Test encode with maximum value
TEST_F(Base32Test, EncodeMaxValue) {
  std::array<char, 12> buffer{};

  // Encode maximum 60-bit value (12 * 5 bits)
  std::uint64_t max_60_bit = (1ULL << 60) - 1;
  Base32::encode(max_60_bit, buffer);

  // All characters should be 'z' (maximum base32 digit)
  for (char ch : buffer) {
    EXPECT_EQ(ch, 'z') << "Maximum value should encode to all 'z'";
  }
}

// Test encode/decode round trip
TEST_F(Base32Test, EncodeDecodeRoundTrip) {
  const std::array<std::uint64_t, 5> test_values = {0u, 31u, 1024u, 1234567u,
                                                    (1ULL << 30) - 1};

  for (auto original_hash : test_values) {
    // Encode
    std::array<char, 13> buffer{};  // Extra space for null terminator
    Base32::encode(original_hash, std::span<char>(buffer.data(), 12));
    buffer[12] = '\0';

    // Decode
    auto [decoded_hash, count] = encoder_.decode(buffer);

    EXPECT_EQ(decoded_hash, original_hash)
        << "Round trip should preserve hash value";
    EXPECT_EQ(count, 12u) << "Should decode all characters";
  }
}

// Test encode with different buffer sizes
TEST_F(Base32Test, EncodeVariableBufferSizes) {
  std::uint64_t hash = 0x1234567u;

  // Test with buffers of different sizes
  for (std::size_t size = 1; size <= 12; ++size) {
    std::vector<char> buffer(size);
    Base32::encode(hash, buffer);

    // Verify all characters are valid
    for (char ch : buffer) {
      std::array<char, 2> test_buffer{ch, '\0'};
      EXPECT_TRUE(encoder_.validate(test_buffer))
          << "Encoded character should be valid for buffer size " << size;
    }
  }
}

// Test decode with known geohash examples
TEST_F(Base32Test, DecodeKnownGeohashes) {
  // Test some real-world geohash examples
  struct TestCase {
    std::string geohash;
    std::size_t expected_count;
  };

  const std::array<TestCase, 4> test_cases = {
      TestCase{.geohash = "ezs42", .expected_count = 5},
      TestCase{.geohash = "u4pruydqqvj", .expected_count = 11},
      TestCase{.geohash = "9q5", .expected_count = 3},
      TestCase{.geohash = "gbsuv7ztqzpt", .expected_count = 12},
  };

  for (const auto& test : test_cases) {
    std::vector<char> buffer(test.geohash.begin(), test.geohash.end());
    buffer.push_back('\0');

    auto [hash, count] = encoder_.decode(buffer);

    EXPECT_EQ(count, test.expected_count)
        << "Geohash '" << test.geohash << "' should decode "
        << test.expected_count << " characters";
    EXPECT_GT(hash, 0u) << "Hash should be non-zero for non-empty geohash";
  }
}

// Test bit shifting in decode
TEST_F(Base32Test, DecodeBitShifting) {
  // "01" should be (0 << 5) | 1 = 1
  std::array<char, 3> buffer1{'0', '1', '\0'};
  auto [hash1, count1] = encoder_.decode(buffer1);
  EXPECT_EQ(hash1, 1u);
  EXPECT_EQ(count1, 2u);

  // "10" should be (1 << 5) | 0 = 32
  std::array<char, 3> buffer2{'1', '0', '\0'};
  auto [hash2, count2] = encoder_.decode(buffer2);
  EXPECT_EQ(hash2, 32u);
  EXPECT_EQ(count2, 2u);
}

// Test constexpr capabilities
TEST(Base32ConstexprTest, ConstexprEncode) {
  // Test that encode can be used in constexpr context
  constexpr auto test_encode = []() -> char {
    std::array<char, 1> buffer{};
    Base32::encode(31u, buffer);
    return buffer[0];
  };

  constexpr char result = test_encode();
  EXPECT_EQ(result, 'z');
}

TEST(Base32ConstexprTest, ConstexprConstructor) {
  // Test constexpr constructor
  constexpr Base32 encoder{};
  (void)encoder;  // Suppress unused variable warning
}

}  // namespace pyinterp::geohash
