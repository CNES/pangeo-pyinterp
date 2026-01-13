// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <array>
#include <format>
#include <string>
#include <string_view>

namespace pyinterp {

/// @brief Format a number of bytes into a human-readable string.
/// @param[in] nbytes Number of bytes.
/// @return Formatted string.
[[nodiscard]] inline auto format_bytes(size_t nbytes) -> std::string {
  constexpr auto units =
      std::array<std::string_view, 5>{"B", "KB", "MB", "GB", "TB"};
  constexpr size_t kBytesPerKilobyte = 1024;
  for (const auto& unit : units) {
    if (nbytes < kBytesPerKilobyte) {
      return std::format("{} {}", nbytes, unit);
    }
    nbytes /= kBytesPerKilobyte;
  }
  return std::format("{} PB", nbytes);
}

}  // namespace pyinterp
