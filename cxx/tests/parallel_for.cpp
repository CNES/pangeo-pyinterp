// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/parallel_for.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
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

  // The worker processing index 16 throws: the exception is propagated back
  // to the caller, whichever thread processed it.
  EXPECT_THROW(parallel_for(
                   size,
                   [](int64_t s, int64_t e) -> void {
                     if (s <= 16 && 16 < e) {
                       throw std::runtime_error("boom");
                     }
                   },
                   4),
               std::runtime_error);
}

// Counts the visits of every index, from any number of threads.
class VisitCounter {
 public:
  explicit VisitCounter(int64_t size) : counts_(static_cast<size_t>(size)) {}

  void operator()(int64_t s, int64_t e) {
    ASSERT_LE(0, s);
    ASSERT_LT(s, e);
    ASSERT_LE(e, static_cast<int64_t>(counts_.size()));
    for (int64_t i = s; i < e; ++i) {
      counts_[static_cast<size_t>(i)].fetch_add(1, std::memory_order_relaxed);
    }
    calls_.fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] auto all_visited_once() const -> bool {
    return std::ranges::all_of(
        counts_, [](const auto& item) -> bool { return item.load() == 1; });
  }

  [[nodiscard]] auto calls() const -> int64_t { return calls_.load(); }

 private:
  std::vector<std::atomic<int>> counts_;
  std::atomic<int64_t> calls_{0};
};

TEST(ParallelFor, EveryIndexVisitedOnce) {
  for (const int64_t size :
       {int64_t{1}, int64_t{2}, int64_t{7}, int64_t{255}, int64_t{256},
        int64_t{257}, int64_t{1000}, int64_t{10007}, int64_t{1000003}}) {
    for (const int64_t num_threads : {int64_t{0}, int64_t{1}, int64_t{2},
                                      int64_t{3}, int64_t{14}, int64_t{64}}) {
      auto counter = VisitCounter(size);
      parallel_for(
          size, [&counter](int64_t s, int64_t e) -> void { counter(s, e); },
          num_threads);
      EXPECT_TRUE(counter.all_visited_once())
          << "size=" << size << " num_threads=" << num_threads;
    }
  }
}

TEST(ParallelFor, EmptyRange) {
  auto calls = 0;
  parallel_for(
      0,
      [&calls](int64_t s, int64_t e) -> void {
        EXPECT_EQ(s, 0);
        EXPECT_EQ(e, 0);
        ++calls;
      },
      4);
  EXPECT_LE(calls, 1);
}

TEST(ParallelFor, SingleThreadMakesOneCall) {
  auto counter = VisitCounter(100000);
  parallel_for(
      100000, [&counter](int64_t s, int64_t e) -> void { counter(s, e); }, 1);
  EXPECT_TRUE(counter.all_visited_once());
  EXPECT_EQ(counter.calls(), 1);
}

TEST(ParallelFor, ChunksAreContiguousAndCoverTheRange) {
  const int64_t size = 100003;
  auto mutex = std::mutex();
  auto ranges = std::vector<std::pair<int64_t, int64_t>>();
  parallel_for(
      size,
      [&](int64_t s, int64_t e) -> void {
        auto lock = std::scoped_lock(mutex);
        ranges.emplace_back(s, e);
      },
      8);
  std::ranges::sort(ranges);
  ASSERT_FALSE(ranges.empty());
  EXPECT_EQ(ranges.front().first, 0);
  EXPECT_EQ(ranges.back().second, size);
  for (size_t ix = 1; ix < ranges.size(); ++ix) {
    EXPECT_EQ(ranges[ix - 1].second, ranges[ix].first);
  }
  // The range is split into more chunks than threads.
  EXPECT_GT(ranges.size(), size_t{8});
}

TEST(ParallelFor, UnevenWorkIsShared) {
  // The first eighth of the range costs far more than the rest: with a static
  // split, the thread given it would do it alone.
  const int64_t size = 4096;
  auto mutex = std::mutex();
  auto per_thread = std::map<std::thread::id, int64_t>();
  parallel_for(
      size,
      [&](int64_t s, int64_t e) -> void {
        for (auto ix = s; ix < e; ++ix) {
          if (ix < size / 8) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
          }
        }
        auto lock = std::scoped_lock(mutex);
        per_thread[std::this_thread::get_id()] += e - s;
      },
      4);
  auto total = int64_t{0};
  for (const auto& [id, count] : per_thread) {
    total += count;
  }
  EXPECT_EQ(total, size);
  // The expensive elements were claimed by more than one thread.
  EXPECT_GT(per_thread.size(), size_t{1});
}

TEST(ParallelFor, ExceptionRethrownAfterJoin) {
  const int64_t size = 100000;
  std::atomic<int64_t> running{0};
  EXPECT_THROW(parallel_for(
                   size,
                   [&running](int64_t s, int64_t e) -> void {
                     running.fetch_add(1);
                     if (s <= size / 2 && size / 2 < e) {
                       running.fetch_sub(1);
                       throw std::runtime_error("boom");
                     }
                     std::this_thread::sleep_for(std::chrono::microseconds(50));
                     running.fetch_sub(1);
                   },
                   4),
               std::runtime_error);
  // Every thread has been joined: no worker is still running.
  EXPECT_EQ(running.load(), 0);
}

TEST(ParallelFor, FirstExceptionWins) {
  // Every chunk throws: exactly one exception reaches the caller.
  EXPECT_THROW(parallel_for(
                   100000,
                   [](int64_t /*s*/, int64_t /*e*/) -> void {
                     throw std::invalid_argument("boom");
                   },
                   8),
               std::invalid_argument);
}

TEST(ParallelFor, ExceptionOnSingleThread) {
  EXPECT_THROW(parallel_for(
                   10,
                   [](int64_t /*s*/, int64_t /*e*/) -> void {
                     throw std::runtime_error("boom");
                   },
                   1),
               std::runtime_error);
}

TEST(ParallelFor, ChunkSize) {
  // Large ranges: num_threads * 32 chunks.
  EXPECT_EQ(detail::determine_chunk_size(1'000'000, 10),
            detail::ceil_div(1'000'000, 320));
  // Medium ranges: the grain.
  EXPECT_EQ(detail::determine_chunk_size(50'000, 10), detail::kGrain);
  // Short ranges: still a few chunks per thread.
  EXPECT_EQ(detail::determine_chunk_size(720, 10),
            detail::ceil_div(720, 10 * detail::kMinChunksPerThread));
  EXPECT_EQ(detail::determine_chunk_size(3, 10), 1);
}

}  // namespace pyinterp
