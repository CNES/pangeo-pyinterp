#pragma once
#include <gsl/gsl_errno.h>

namespace pyinterp {
namespace detail {
namespace gsl {

/// GSL Error handler
///
/// @param reason The reason for the error
/// @param code Error number
void error_handler(const char* reason, const char* /*unused*/, int /*unused*/,
                   int code);

/// Sets the error handler for the GSL library routines.
void set_error_handler();

}  // namespace gsl
}  // namespace detail
}  // namespace pyinterp
