// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace pyinterp::geohash {

// Encoding encapsulates an encoding defined by a given base32 alphabet.
class Base32 {
 public:
  // Default constructor
  Base32() noexcept {
    decode_.fill(Base32::kInvalid_);
    for (size_t ix = 0; ix < encode_.size(); ++ix) {
      decode_[static_cast<unsigned char>(encode_[ix])] = static_cast<char>(ix);
    }
  }

  // Returns true if the buffer contains a valid definition of this encoding.
  [[nodiscard]] constexpr auto validate(const char *hash,
                                        const size_t count) const -> bool {
    const auto *end = hash + count;
    while (hash != end && *hash != 0) {
      if (!validate_byte(*(hash++))) {
        return false;
      }
    }
    return true;
  }

  // Returns the string decoded into bits of a 64-bit word and the the number of
  // characters other than the null character.
  [[nodiscard]] constexpr auto decode(const char *const buffer,
                                      const size_t count) const
      -> std::tuple<uint64_t, uint32_t> {
    auto hash = static_cast<uint64_t>(0);
    const auto *it = buffer;
    while (it != buffer + count && *it != 0) {
      if (!validate_byte(*it)) {
        throw std::invalid_argument("Invalid character in hash: " +
                                    std::string(buffer, count));
      }
      hash = (hash << 5U) | static_cast<uint64_t>(
                                decode_[static_cast<unsigned char>(*(it++))]);
    }
    return std::make_tuple(hash, static_cast<uint32_t>(it - buffer));
  }

  // Encode bits of 64-bit word into a string.
  constexpr static auto encode(uint64_t hash, char *const buffer,
                               const size_t count) -> void {
    auto *it = buffer + count - 1;
    while (it >= buffer) {
      *(it--) = encode_[hash & 0x1fU];
      hash >>= 5U;
    }
  }

 private:
  static const char kInvalid_;
  static const std::array<char, 32> encode_;
  std::array<char, std::numeric_limits<uint8_t>::max() + 1> decode_{};

  // Reports whether byte is part of the encoding.
  [[nodiscard]] constexpr auto validate_byte(const char byte) const -> bool {
    return decode_[static_cast<unsigned char>(byte)] != Base32::kInvalid_;
  }
};

}  // namespace pyinterp::geohash
