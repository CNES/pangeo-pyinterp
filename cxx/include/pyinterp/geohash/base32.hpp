// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once
#include <array>
#include <cstdint>
#include <format>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace pyinterp::geohash {

/// @brief Encoding encapsulates an encoding defined by a given base32 alphabet.
class Base32 {
 public:
  /// Default constructor.
  constexpr Base32() noexcept : decode_{} {
    decode_.fill(kInvalid);
    for (std::size_t ix = 0; ix < kEncode.size(); ++ix) {
      decode_[static_cast<unsigned char>(kEncode[ix])] = static_cast<char>(ix);
    }
  }

  /// @brief Validates that all characters of the hash are part of the encoding.
  /// @param[in] hash Geohash to validate.
  [[nodiscard]] constexpr auto validate(
      std::span<const char> hash) const noexcept -> bool {
    for (const auto byte : hash) {
      if (byte == '\0') {
        break;
      }
      if (!validate_byte(byte)) {
        return false;
      }
    }
    return true;
  }

  /// @brief Decodes the string into bits of a 64-bit word and returns the
  /// number of characters other than the null character.
  /// @param[in] buffer Buffer containing the geohash to decode.
  /// @return A tuple containing the decoded 64-bit word and the number of
  /// characters.
  [[nodiscard]] constexpr auto decode(std::span<const char> buffer) const
      -> std::tuple<std::uint64_t, std::uint32_t> {
    std::uint64_t hash{};
    std::uint32_t count{};

    for (const auto byte : buffer) {
      if (byte == '\0') {
        break;
      }
      if (!validate_byte(byte)) {
        throw std::invalid_argument(
            std::format("Invalid character in hash: {}",
                        std::string_view{buffer.data(), buffer.size()}));
      }
      hash = (hash << 5U) | static_cast<std::uint64_t>(
                                decode_[static_cast<unsigned char>(byte)]);
      ++count;
    }
    return {hash, count};
  }

  /// @brief Encodes the 64-bit word into a string using the base32 encoding.
  /// @param[in] hash 64-bit word to encode.
  /// @param[out] buffer Buffer to store the encoded string.
  static constexpr auto encode(std::uint64_t hash,
                               std::span<char> buffer) noexcept -> void {
    for (char& it : std::ranges::reverse_view(buffer)) {
      it = kEncode[hash & 0x1FU];
      hash >>= 5U;
    }
  }

 private:
  /// Invalid character in the decoding table.
  static constexpr char kInvalid{static_cast<char>(0xFF)};
  /// Base32 encoding table.
  static constexpr std::array<char, 32> kEncode{
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'b',
      'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p',
      'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
  /// Decoding table.
  std::array<char, std::numeric_limits<std::uint8_t>::max() + 1> decode_{};

  /// @brief Validates that the byte is part of the encoding.
  /// @param[in] byte Byte to validate.
  [[nodiscard]] constexpr auto validate_byte(const char byte) const noexcept
      -> bool {
    return decode_[static_cast<unsigned char>(byte)] != kInvalid;
  }
};

}  // namespace pyinterp::geohash
