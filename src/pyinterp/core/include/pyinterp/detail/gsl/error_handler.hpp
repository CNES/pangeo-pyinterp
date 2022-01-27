// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <gsl/gsl_errno.h>

namespace pyinterp::detail::gsl {

/// GSL Error handler
///
/// @param reason The reason for the error
/// @param code Error number
void error_handler(const char *reason, const char * /*unused*/, int /*unused*/,
                   int code);

/// Sets the error handler for the GSL library routines.
void set_error_handler();

}  // namespace pyinterp::detail::gsl
