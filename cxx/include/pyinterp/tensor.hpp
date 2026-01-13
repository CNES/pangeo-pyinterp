// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cstdint>
#include <format>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace pyinterp::tensor {

/// @brief Alias of vector of dimensions
using Shape = std::vector<size_t>;

/// @brief Alias of vector of indexes
using VectorIndex = std::vector<int64_t>;

/// Compute properties for dimension reduction
struct ReducedProperties {
  Shape shape;
  VectorIndex strides;
  VectorIndex adjusted_strides;
};

/// @brief Compute properties for dimension reduction for a C-contiguous array
/// @param[in] input_shape Shape of the input array
/// @param[in] input_strides Strides of the input array
/// @param[in] axis Axis indices to reduce over
/// @returns Computed reduced properties
[[nodiscard]] inline auto compute_reduced_properties(
    const Shape& input_shape, const VectorIndex& input_strides,
    const VectorIndex& axis) -> ReducedProperties {
  auto ndim = static_cast<int64_t>(input_shape.size());

  // Copy input strides
  auto strides = input_strides;

  // Prepare output structure
  VectorIndex adjusted_strides(ndim);
  Shape shape;
  shape.reserve(ndim);

  // Identify which axes to reduce
  std::vector<bool> is_reduced(ndim, false);
  for (const auto& ax : axis) {
    if (ax >= 0 && ax < ndim) {
      is_reduced[ax] = true;
    }
  }

  // Compute Reduced Shape (Forward pass)
  for (int64_t ix = 0; ix < ndim; ++ix) {
    if (!is_reduced[ix]) {
      shape.push_back(input_shape[ix]);
    }
  }

  // Compute Adjusted Strides (Backward pass)
  // These are the strides of the *resulting* reduced array, mapped back to the
  // corresponding input dimensions.
  // - If a dimension is reduced, its contribution to the output pointer offset
  //   is 0.
  // - If a dimension is kept, its stride is determined by the cumulative size
  //   of the *subsequent* kept dimensions (standard C-contiguous logic).
  int64_t current_stride = 1;

  for (int64_t ix = ndim - 1; ix >= 0; --ix) {
    if (is_reduced[ix]) {
      // Dimension is collapsed; iterating over it does not move the output
      // pointer.
      adjusted_strides[ix] = 0;
    } else {
      // Dimension is preserved; assign current cumulative stride.
      adjusted_strides[ix] = current_stride;
      // Update cumulative stride using the size of this preserved dimension.
      current_stride *= input_shape[ix];
    }
  }

  return {.shape = std::move(shape),
          .strides = std::move(strides),
          .adjusted_strides = std::move(adjusted_strides)};
}

/// @brief Unravel flat index to multi-dimensional indices
/// @param[in] flat_idx Flat index to unravel
/// @param[in] strides Strides for each dimension
/// @param[out] indices Output multi-dimensional indices
constexpr void unravel_index(int64_t flat_idx, const VectorIndex& strides,
                             VectorIndex& indices) noexcept {
  for (auto [indice_item, stride_item] : std::views::zip(indices, strides)) {
    indice_item = flat_idx / stride_item;
    flat_idx -= indice_item * stride_item;
  }
}

/// @brief Compute flat offset from multi-dimensional indices using strides
/// @param[in] indices Multi-dimensional indices
/// @param[in] strides Element strides for each dimension
[[nodiscard]] constexpr auto compute_offset(const VectorIndex& indices,
                                            const VectorIndex& strides) noexcept
    -> int64_t {
  int64_t offset = 0;
  for (size_t ix = 0; ix < indices.size(); ++ix) {
    offset += indices[ix] * strides[ix];
  }
  return offset;
}

/// @brief Increment multi-dimensional indices (row-major order)
/// @param[in,out] indices Multi-dimensional indices to increment
/// @param[in] shape Shape of the array
constexpr void increment_indices(VectorIndex& indices,
                                 const Shape& shape) noexcept {
  const auto ndim = static_cast<int64_t>(shape.size());
  for (auto ix = ndim - 1; ix >= 0; --ix) {
    ++indices[ix];
    if (indices[ix] < static_cast<int64_t>(shape[ix])) {
      return;
    }
    indices[ix] = 0;
  }
}

/// @brief Validate and convert axis.
/// In Python, negative axis values are allowed to index from the end.
/// This function converts negative axis values to their positive equivalents
/// and checks that all axis values are within valid bounds.
/// @param[in] axis axis values (can be negative)
/// @param[in] ndim Number of dimensions of the array
/// @returns Converted axis values
[[nodiscard]]
inline auto validate_and_convert_axis(const VectorIndex& axis,
                                      const size_t ndim) -> VectorIndex {
  auto result = VectorIndex{};
  result.reserve(ndim);
  for (auto ax : axis) {
    // Handle negative axis (Python convention)
    if (ax < 0) {
      ax += static_cast<int64_t>(ndim);
    }
    if (ax < 0 || ax >= static_cast<int64_t>(ndim)) {
      throw std::out_of_range(std::format(
          "axis {} is out of bounds for array of dimension {}", ax, ndim));
    }
    result.push_back(ax);
  }
  return result;
}

}  // namespace pyinterp::tensor
