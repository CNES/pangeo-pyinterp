#pragma once

namespace pyinterp::fill {

/// Type of first guess grid.
enum FirstGuess {
  kZero,          //!< Use 0.0 as an initial guess
  kZonalAverage,  //!< Use zonal average in x direction
};

}  // namespace pyinterp::fill
