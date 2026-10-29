// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once
#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace pyinterp {

/// Concept for worker functions that accept a range of indices
template <typename F>
concept WorkerFunction = requires(F f, int64_t start, int64_t end) {
  { f(start, end) } -> std::same_as<void>;
};

namespace detail {

/// Cache-line size used to pad the shared index against false sharing.
///
/// `std::hardware_destructive_interference_size` is not used: GCC warns
/// (`-Winterference-size`) that its value depends on the compiler version and
/// the `-mtune`/`-mcpu` flags. 64 bytes covers x86-64 and the common ARM64
/// targets; Apple silicon uses 128-byte lines, hence 128.
inline constexpr std::size_t kCacheLineSize = 128;

/// Number of chunks per thread the range is split into, so that a thread left
/// with an expensive chunk (points near undefined values, a dense area of an
/// R*-tree, an efficiency core) does not finish alone while the others wait.
inline constexpr int64_t kChunksPerThread = 32;

/// Minimum number of elements in a chunk. Several workers pay a set-up cost
/// at each call (the windowed interpolators build a cache and an interpolator,
/// LOESS allocates its workspace, the bicubic cache starts empty): smaller
/// chunks would spend more time setting up than computing.
inline constexpr int64_t kGrain = 256;

/// Minimum number of chunks per thread. When the range is too short to fill
/// `kChunksPerThread` chunks of `kGrain` elements, the grain gives way so that
/// every thread still claims a few chunks: an element may then be a whole row
/// of a grid (LOESS, zonal averages, GeoHash masks) rather than a point.
inline constexpr int64_t kMinChunksPerThread = 4;

/// Ceiling of `a / b` for positive values, without overflow.
[[nodiscard]] constexpr auto ceil_div(int64_t a, int64_t b) noexcept
    -> int64_t {
  return (a / b) + static_cast<int64_t>((a % b) != 0);
}

/// Number of elements of the chunks the range `[0, size)` is split into.
///
/// @param[in] size Total number of elements (> 0).
/// @param[in] num_threads Number of threads sharing the work (> 0).
/// @return Chunk size, at least 1.
[[nodiscard]] constexpr auto determine_chunk_size(int64_t size,
                                                  int64_t num_threads) noexcept
    -> int64_t {
  const auto fine = ceil_div(size, num_threads * kChunksPerThread);
  const auto coarse = ceil_div(size, num_threads * kMinChunksPerThread);
  return std::max(std::min(std::max(fine, kGrain), coarse), int64_t{1});
}

/// Runs `worker` over `[0, size)` by chunks of `chunk_size` elements, claimed
/// one at a time from a shared index by the calling thread and up to
/// `num_threads - 1` helper threads.
///
/// A helper that cannot be started (`std::thread` throws `std::system_error`
/// when a thread limit is reached) stops the starting, and the work goes on
/// with the threads already running, the caller at least: the chunks being
/// claimed from a shared index, the worker is called over the same ranges.
/// No thread is left joinable while an exception unwinds.
///
/// @throws The first exception thrown by the worker, once every helper has
/// been joined.
template <typename Worker>
void run_chunks(Worker& worker, const int64_t size, const int64_t num_threads,
                const int64_t chunk_size) {
  alignas(kCacheLineSize) std::atomic<int64_t> next{0};
  auto first_exception = std::exception_ptr(nullptr);
  auto mutex = std::mutex();

  // What every thread runs, the caller included.
  const auto body = [&]() noexcept -> void {
    try {
      while (true) {
        const auto start =
            next.fetch_add(chunk_size, std::memory_order_relaxed);
        if (start >= size) {
          break;
        }
        worker(start, start + std::min(chunk_size, size - start));
      }
    } catch (...) {
      // The chunks not yet claimed are abandoned: the call fails anyway.
      next.store(size, std::memory_order_relaxed);
      auto lock = std::scoped_lock(mutex);
      if (!first_exception) {
        first_exception = std::current_exception();
      }
    }
  };

  std::vector<std::thread> helpers;
  helpers.reserve(static_cast<std::size_t>(num_threads - 1));
  for (int64_t ix = 1; ix < num_threads; ++ix) {
    try {
      helpers.emplace_back(body);
    } catch (...) {
      break;  // Thread limit reached: go on with the threads running.
    }
  }

  body();

  for (auto& helper : helpers) {
    helper.join();
  }

  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
}

}  // namespace detail

/// @brief Parallel for loop utility
///
/// The range `[0, size)` is split into contiguous chunks that the threads
/// claim one at a time, so that a thread given cheap elements takes more of
/// them. The calling thread takes part in the work; no thread is started when
/// the range amounts to a single chunk. The worker may therefore be called
/// several times by the same thread, and must only write to the indexes of
/// the range it receives.
///
/// @tparam Worker Type of the worker function
/// @param[in] size Total number of iterations
/// @param[in] worker Worker function accepting (start, end) indices
/// @param[in] num_threads Number of threads to use (0 for hardware concurrency)
/// @throws The first exception thrown by the worker, once every thread has
/// finished.
template <WorkerFunction Worker>
void parallel_for(int64_t size, Worker&& worker, int64_t num_threads = 0) {
  if (num_threads == 0) {
    num_threads = std::max<int64_t>(std::thread::hardware_concurrency(), 1);
  }
  if (num_threads <= 1 || size <= 1) {
    worker(0, size);
    return;
  }

  const auto chunk_size = detail::determine_chunk_size(size, num_threads);

  // Never start a thread that would find no chunk left to claim.
  num_threads = std::min(num_threads, detail::ceil_div(size, chunk_size));
  if (num_threads == 1) {
    worker(0, size);
    return;
  }

  detail::run_chunks(worker, size, num_threads, chunk_size);
}

}  // namespace pyinterp
