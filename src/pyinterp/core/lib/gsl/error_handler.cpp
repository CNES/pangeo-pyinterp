// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/detail/gsl/error_handler.hpp"

#include <sstream>

namespace pyinterp::detail::gsl {

void error_handler(const char *reason, const char * /*unused*/, int /*unused*/,
                   int code) {
  std::ostringstream error;
  error << reason << " (GSL error #" << code << ")";
  throw std::runtime_error(error.str());
}

void set_error_handler() { gsl_set_error_handler(error_handler); }

}  // namespace pyinterp::detail::gsl
