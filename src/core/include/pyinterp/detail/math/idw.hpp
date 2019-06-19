#pragma once
#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/thread.hpp"
#include <Eigen/Core>

namespace pyinterp {
namespace detail {
namespace math {

/// Interpolation of the value by Inverse Distance Weighting.
template <typename T>
std::tuple<T, uint32_t> inverse_distance_weigthing_(
    const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>& distance,
    const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>& value,
    const double radius, const size_t p = 2) {
  uint32_t neighbors = 0;
  T result = T(0);
  T total_weight = T(0);

  for (auto ix = 0; ix < distance.size(); ++ix) {
    if (distance(ix) < 1e-6) {
      // If the user has requested a grid point, the mesh value is returned.
      return std::make_tuple(value(ix), distance.size());
    }

    if (distance(ix) <= radius) {
      // If the neighbor found is within an acceptable radius it can be taken
      // into account in the calculation.
      T wk = T(1 / std::pow(distance(ix), p));
      total_weight += wk;
      result += value(ix) * wk;
      ++neighbors;
    }
  }
  // Finally the interpolated value is returned if there are selected points
  // otherwise one returns an undefined value.
  return total_weight != 0
             ? std::make_tuple(static_cast<T>(result / total_weight), neighbors)
             : std::make_tuple(std::numeric_limits<T>::quiet_NaN(),
                               static_cast<uint32_t>(0));
}

/// Interpolation of the value by Inverse Distance Weighting.
///
/// @param distance Distance between the nearest point and the point of interest
/// @param value Value of nearest points
/// @param radius The maximum radius of the search (m).
/// @param p Power parameters.
/// @return a tuple containing the interpolated value and the number of
/// neighbors used in the calculation.
/// @param num_threads The number of threads to use for the computation
template <typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>>
inverse_distance_weigthing(
    const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&
        distance,
    const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& value,
    const double radius = std::numeric_limits<double>::max(),
    const size_t p = 2, const size_t num_threads = 0) {
  check_eigen_shape("distance", distance, "value", value);

  auto size = distance.rows();
  auto interp = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(size);
  auto samples = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>(size);

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  // Dispatch calculation on defined cores
  dispatch(
      [&](const size_t start, const size_t stop) {
        for (auto ix = start; ix < stop; ++ix) {
          std::tie(interp(ix), samples(ix)) = inverse_distance_weigthing_(
              distance.row(ix), value.row(ix), radius, p);
        }
      },
      size, num_threads);

  if (except != nullptr) {
    std::rethrow_exception(except);
  }
  return std::make_tuple(interp, samples);
}

}  // namespace math
}  // namespace detail
}  // namespace pyinterp