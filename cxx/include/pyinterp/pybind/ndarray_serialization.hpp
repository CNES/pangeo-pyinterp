// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::pybind {

using NanobindArray1DUInt8 =
    nanobind::ndarray<nanobind::numpy, uint8_t, nanobind::ndim<1>>;

/// @brief Helper to convert ndarray to SerializationReader (zero-copy with
/// capsule)
/// @param[in] array Input array
/// @return reader instance
inline auto reader_from_ndarray(const NanobindArray1DUInt8& array)
    -> serialization::Reader {
  auto ptr = reinterpret_cast<const std::byte*>(array.data());
  auto size = array.size();
  return {ptr, size, NanobindArray1DUInt8(array)};
}

/// @brief Helper to convert SerializationWriter to ndarray (zero-copy with
/// capsule)
/// @param[in] writer SerializationWriter instance
/// @return ndarray instance
inline auto writer_to_ndarray(serialization::Writer&& writer)
    -> NanobindArray1DUInt8 {
  auto data = std::move(writer).release();
  auto size = data.size();

  auto ptr = std::make_unique<std::vector<std::byte>>(std::move(data));
  auto* data_ptr = ptr->data();

  // Create capsule with deleter that owns the vector
  nanobind::capsule capsule(ptr.get(), [](void* data) noexcept {
    delete static_cast<std::vector<std::byte>*>(data);
  });
  ptr.release();

  return NanobindArray1DUInt8(reinterpret_cast<uint8_t*>(data_ptr), {size},
                              capsule);
}

}  // namespace pyinterp::pybind
