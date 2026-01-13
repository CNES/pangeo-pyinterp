// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <exception>
#include <ranges>
#include <thread>
#include <vector>

namespace pyinterp {

/// Concept for worker functions that accept a range of indices
template <typename F>
concept WorkerFunction = requires(F f, int64_t start, int64_t end) {
  { f(start, end) } -> std::same_as<void>;
};

/// @brief Parallel for loop utility
///
/// @tparam Worker Type of the worker function
/// @param[in] size Total number of iterations
/// @param[in] worker Worker function accepting (start, end) indices
/// @param[in] num_threads Number of threads to use (0 for hardware concurrency)
template <WorkerFunction Worker>
void parallel_for(int64_t size, Worker&& worker, int64_t num_threads = 0) {
  if (num_threads == 1) {
    worker(0, size);
    return;
  }

  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
  }

  num_threads = std::min(num_threads, size);
  if (num_threads <= 1) {
    worker(0, size);
    return;
  }

  const int64_t chunk_size = size / num_threads;
  const int64_t remainder = size % num_threads;

  std::vector<std::jthread> threads;
  std::vector<std::exception_ptr> exceptions(num_threads);
  threads.reserve(num_threads);

  int64_t start = 0;
  for (int64_t i : std::views::iota(int64_t{0}, num_threads)) {
    const int64_t end = start + chunk_size + (i < remainder ? 1 : 0);
    threads.emplace_back(
        [&worker, &exceptions, i](int64_t s, int64_t e) {
          try {
            worker(s, e);
          } catch (...) {
            exceptions[i] = std::current_exception();
          }
        },
        start, end);
    start = end;
  }

  threads.clear();

  // Rethrow first exception encountered
  for (const auto& eptr : exceptions) {
    if (eptr) {
      std::rethrow_exception(eptr);
    }
  }
}

}  // namespace pyinterp
