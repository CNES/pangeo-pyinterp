// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/parallel_for.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace pyinterp {

TEST(ParallelFor, SingleThread) {
  const int64_t size = 1000;
  std::vector<int> data(size, 0);

  parallel_for(
      size,
      [&data](int64_t s, int64_t e) -> void {
        for (int64_t i = s; i < e; ++i) {
          data[static_cast<size_t>(i)] = 1;
        }
      },
      1);

  EXPECT_EQ(std::ranges::count(data, 1), size);
}

TEST(ParallelFor, MultipleThreads) {
  const int64_t size = 10007;
  std::vector<int> data(size, 0);

  parallel_for(
      size,
      [&data](int64_t s, int64_t e) -> void {
        for (int64_t i = s; i < e; ++i) {
          data[static_cast<size_t>(i)] = 1;
        }
      },
      4);

  EXPECT_EQ(std::ranges::count(data, 1), size);
}

TEST(ParallelFor, NumThreadsZeroUsesHardwareConcurrency) {
  const int64_t size = 1024;
  std::vector<int> data(size, 0);

  parallel_for(
      size,
      [&data](int64_t s, int64_t e) -> void {
        for (int64_t i = s; i < e; ++i) {
          data[static_cast<size_t>(i)] = 1;
        }
      },
      0);

  EXPECT_EQ(std::ranges::count(data, 1), size);
}

TEST(ParallelFor, NumThreadsGreaterThanSize) {
  const int64_t size = 7;
  std::vector<int> data(size, 0);

  parallel_for(
      size,
      [&data](int64_t s, int64_t e) -> void {
        for (int64_t i = s; i < e; ++i) {
          data[static_cast<size_t>(i)] = 1;
        }
      },
      100);

  EXPECT_EQ(std::ranges::count(data, 1), size);
}

TEST(ParallelFor, ExceptionRethrown) {
  const int64_t size = 32;

  // With size=32 and num_threads=4 the chunks are [0,8),[8,16),[16,24),[24,32)
  // We throw from the worker that receives start==16 to ensure an exception
  // occurs inside a thread and is propagated back to the caller.
  EXPECT_THROW(parallel_for(
                   size,
                   [](int64_t s, int64_t e) -> void {
                     if (s == 16) {
                       throw std::runtime_error("boom");
                     }
                   },
                   4),
               std::runtime_error);
}

}  // namespace pyinterp
