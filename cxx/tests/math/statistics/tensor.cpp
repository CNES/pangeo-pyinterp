// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/tensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace pyinterp::tensor {

// Reference: the loop on compute_offset and increment_indices.
static auto reference_offsets(const Shape& shape,
                              const std::vector<VectorIndex>& strides)
    -> std::vector<std::vector<int64_t>> {
  const auto total =
      std::ranges::fold_left(shape, size_t{1}, std::multiplies{});
  auto indices = VectorIndex(shape.size(), 0);
  auto result = std::vector<std::vector<int64_t>>();
  for (size_t ix = 0; ix < total; ++ix) {
    auto offsets = std::vector<int64_t>();
    for (const auto& item : strides) {
      offsets.push_back(compute_offset(indices, item));
    }
    result.push_back(offsets);
    increment_indices(indices, shape);
  }
  return result;
}

TEST(Tensor, ForEachOffsetMatchesReference) {
  // C-contiguous values, transposed weights, reduced output along axis 1
  const auto shape = Shape{3, 4, 5};
  const auto values = VectorIndex{20, 5, 1};
  const auto weights = VectorIndex{1, 3, 12};
  const auto output = VectorIndex{5, 0, 1};

  auto visited = std::vector<std::vector<int64_t>>();
  for_each_offset<3>(shape, {&values, &weights, &output},
                     [&](const std::array<int64_t, 3>& offsets) {
                       visited.push_back({offsets[0], offsets[1], offsets[2]});
                     });
  EXPECT_EQ(visited, reference_offsets(shape, {values, weights, output}));
}

TEST(Tensor, ForEachOffsetEdgeCases) {
  const auto strides1 = VectorIndex{1};
  auto count = 0;
  // One dimension
  for_each_offset<1>(Shape{7}, {&strides1},
                     [&](const std::array<int64_t, 1>& offsets) {
                       EXPECT_EQ(offsets[0], count++);
                     });
  EXPECT_EQ(count, 7);

  // Empty array: no visit
  count = 0;
  const auto strides2 = VectorIndex{3, 1};
  for_each_offset<1>(Shape{0, 3}, {&strides2},
                     [&](const std::array<int64_t, 1>&) { ++count; });
  EXPECT_EQ(count, 0);

  // Zero-dimensional array: a single element at offset 0
  const auto strides0 = VectorIndex{};
  for_each_offset<1>(Shape{}, {&strides0},
                     [&](const std::array<int64_t, 1>& offsets) {
                       EXPECT_EQ(offsets[0], 0);
                       ++count;
                     });
  EXPECT_EQ(count, 1);
}

}  // namespace pyinterp::tensor
