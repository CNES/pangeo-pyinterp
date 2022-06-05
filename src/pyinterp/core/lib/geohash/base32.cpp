// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geohash/base32.hpp"

namespace pyinterp::geohash {

// Invalid character in an encoding
const char Base32::kInvalid_ = std::numeric_limits<char>::max();

// Encoding characters
const std::array<char, 32> Base32::encode_ =
    std::array<char, 32>{{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'b',
                          'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p',
                          'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}};
}  // namespace pyinterp::geohash
