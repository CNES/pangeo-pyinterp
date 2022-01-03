// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/detail/thread.hpp"

#include <gtest/gtest.h>

TEST(thread, dispatch) {
  std::vector<double> src(4096);
  std::vector<double> dst(4096);
  for (auto ix = 0; ix < 4096; ++ix) {
    src[ix] = ix;
  }

  auto foo = [&dst, &src](size_t start, size_t stop) {
    for (auto ix = start; ix < stop; ++ix) {
      dst[ix] = src[ix];
    }
  };

  pyinterp::detail::dispatch(foo, src.size(), 0);

  for (auto ix = 0; ix < 4096; ++ix) {
    EXPECT_EQ(src[ix], dst[ix]);
  }
}
