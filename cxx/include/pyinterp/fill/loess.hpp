#pragma once

#include "pyinterp/detail/thread.hpp"
#include "pyinterp/fill/utils.hpp"
#include "pyinterp/grid.hpp"

namespace pyinterp::fill {

/// Type of values processed by the Loess filter.
enum ValueType {
  kUndefined,  //!< Undefined values (fill undefined values)
  kDefined,    //!< Defined values (smooth values)
  kAll         //!< Smooth and fill values
};

/// Fills undefined values using a locally weighted regression function or
/// LOESS. The weight function used for LOESS is the tri-cube weight
/// function, w(x)=(1-|d|^{3})^{3}
///
/// @param grid Grid Function on a uniform 2-dimensional grid to be filled.
/// @param nx Number of points of the half-window to be taken into account
/// along the longitude axis.
/// @param nx Number of points of the half-window to be taken into account
/// along the latitude axis.
/// @param value_type Type of values processed by the filter
/// @param num_threads The number of threads to use for the computation. If
/// 0 all CPUs are used. If 1 is given, no parallel computing code is used
/// at all, which is useful for debugging.
/// @return The grid will have all the NaN filled with extrapolated values.
template <typename Type>
auto loess(const Grid2D<Type> &grid, const uint32_t nx, const uint32_t ny,
           const ValueType value_type, const size_t num_threads)
    -> pybind11::array_t<Type> {
  check_windows_size("nx", nx, "ny", ny);
  auto result = pybind11::array_t<Type>(
      pybind11::array::ShapeContainer{grid.x()->size(), grid.y()->size()});
  auto _result = result.template mutable_unchecked<2>();

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  auto worker = [&](const size_t start, const size_t end) {
    try {
      // Access to the shared pointer outside the loop to avoid data races
      const auto &x_axis = *grid.x();
      const auto &y_axis = *grid.y();
      auto x_frame = std::vector<int64_t>(nx * 2 + 1);
      auto y_frame = std::vector<int64_t>(ny * 2 + 1);

      for (size_t ix = start; ix < end; ++ix) {
        auto x = x_axis(ix);

        // We retrieve the indexes framing the current value.
        frame_index(ix, x_axis.size(), x_axis.is_angle(), x_frame);

        // Read the first value of the calculated window.
        const auto x0 = x_axis(x_frame[0]);

        // The current value is normalized to the first value in the
        // window.
        if (x_axis.is_angle()) {
          x = detail::math::normalize_angle(x, x0, 360.0);
        }

        for (int64_t iy = 0; iy < y_axis.size(); ++iy) {
          auto z = grid.value(ix, iy);

          // If the current value is masked.
          const auto undefined = std::isnan(z);
          if (value_type == kAll || (value_type == kDefined && !undefined) ||
              (value_type == kUndefined && undefined)) {
            auto y = y_axis(iy);

            // We retrieve the indexes framing the current value.
            frame_index(iy, y_axis.size(), false, y_frame);

            // Initialization of values to calculate the extrapolated
            // value.
            auto value = Type(0);
            auto weight = Type(0);

            // For all the coordinates of the frame.
            for (auto wx : x_frame) {
              auto xi = x_axis(wx);

              // We normalize the window's coordinates to its first value.
              if (x_axis.is_angle()) {
                xi = detail::math::normalize_angle(xi, x0, 360.0);
              }

              for (auto wy : y_frame) {
                auto zi = grid.value(wx, wy);

                // If the value is not masked, its weight is calculated from
                // the tri-cube weight function
                if (!std::isnan(zi)) {
                  const auto power = 3.0;
                  auto d =
                      std::sqrt(detail::math::sqr(((xi - x)) / nx) +
                                detail::math::sqr(((y_axis(wy) - y)) / ny));
                  auto wi = d <= 1 ? std::pow((1.0 - std::pow(d, power)), power)
                                   : 0.0;
                  value += static_cast<Type>(wi * zi);
                  weight += static_cast<Type>(wi);
                }
              }
            }

            // Finally, we calculate the extrapolated value if possible,
            // otherwise we will recopy the masked original value.
            if (weight != 0) {
              z = value / weight;
            }
          }
          _result(ix, iy) = z;
        }
      }
    } catch (...) {
      except = std::current_exception();
    }
  };

  {
    pybind11::gil_scoped_release release;
    detail::dispatch(worker, grid.x()->size(), num_threads);
  }
  return result;
}

template <typename Type, typename GridType, typename... Index>
auto loess_(const GridType &grid, const uint32_t nx, const uint32_t ny,
            const ValueType value_type, const Axis<double> &x_axis,
            const Axis<double> &y_axis, const std::vector<int64_t> &x_frame,
            std::vector<int64_t> &y_frame, const double x0, const double x,
            const int64_t ix, const int64_t iy, Index &&...index) -> Type {
  auto z = grid.value(ix, iy, index...);

  // If the current value is masked.
  const auto undefined = std::isnan(z);
  if (value_type == kAll || (value_type == kDefined && !undefined) ||
      (value_type == kUndefined && undefined)) {
    auto y = y_axis(iy);

    // We retrieve the indexes framing the current value.
    frame_index(iy, y_axis.size(), false, y_frame);

    // Initialization of values to calculate the extrapolated
    // value.
    auto value = Type(0);
    auto weight = Type(0);

    // For all the coordinates of the frame.
    for (auto wx : x_frame) {
      auto xi = x_axis(wx);

      // We normalize the window's coordinates to its first value.
      if (x_axis.is_angle()) {
        xi = detail::math::normalize_angle(xi, x0, 360.0);
      }

      for (auto wy : y_frame) {
        auto zi = grid.value(wx, wy, index...);

        // If the value is not masked, its weight is calculated
        // from the tri-cube weight function
        if (!std::isnan(zi)) {
          const auto power = 3.0;
          auto d = std::sqrt(detail::math::sqr(((xi - x)) / nx) +
                             detail::math::sqr(((y_axis(wy) - y)) / ny));
          auto wi = d <= 1 ? std::pow((1.0 - std::pow(d, power)), power) : 0.0;
          value += static_cast<Type>(wi * zi);
          weight += static_cast<Type>(wi);
        }
      }
    }
    // Finally, we calculate the extrapolated value if possible,
    // otherwise we will recopy the masked original value.
    if (weight != 0) {
      z = value / weight;
    }
  }
  return z;
}

template <typename Type, typename AxisType>
auto loess(const Grid3D<Type, AxisType> &grid, const uint32_t nx,
           const uint32_t ny, const ValueType value_type,
           const size_t num_threads) -> pybind11::array_t<Type> {
  check_windows_size("nx", nx, "ny", ny);
  auto result = pybind11::array_t<Type>(pybind11::array::ShapeContainer{
      grid.x()->size(), grid.y()->size(), grid.z()->size()});
  auto _result = result.template mutable_unchecked<3>();

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  auto worker = [&](const size_t start, const size_t end) {
    try {
      // Access to the shared pointer outside the loop to avoid data races
      const auto &x_axis = *grid.x();
      const auto &y_axis = *grid.y();
      auto x_frame = std::vector<int64_t>(nx * 2 + 1);
      auto y_frame = std::vector<int64_t>(ny * 2 + 1);

      for (size_t iz = start; iz < end; ++iz) {
        for (int64_t ix = 0; ix < x_axis.size(); ++ix) {
          auto x = x_axis(ix);

          // We retrieve the indexes framing the current value.
          frame_index(ix, x_axis.size(), x_axis.is_angle(), x_frame);

          // Read the first value of the calculated window.
          const auto x0 = x_axis(x_frame[0]);

          // The current value is normalized to the first value in the
          // window.
          if (x_axis.is_angle()) {
            x = detail::math::normalize_angle(x, x0, 360.0);
          }

          for (int64_t iy = 0; iy < y_axis.size(); ++iy) {
            _result(ix, iy, iz) = loess_<Type, Grid3D<Type, AxisType>>(
                grid, nx, ny, value_type, x_axis, y_axis, x_frame, y_frame, x0,
                x, ix, iy, iz);
          }
        }
      }
    } catch (...) {
      except = std::current_exception();
    }
  };

  {
    pybind11::gil_scoped_release release;
    detail::dispatch(worker, grid.z()->size(), num_threads);
  }
  return result;
}

template <typename Type, typename AxisType>
auto loess(const Grid4D<Type, AxisType> &grid, const uint32_t nx,
           const uint32_t ny, const ValueType value_type,
           const size_t num_threads) -> pybind11::array_t<Type> {
  check_windows_size("nx", nx, "ny", ny);
  auto result = pybind11::array_t<Type>(pybind11::array::ShapeContainer{
      grid.x()->size(), grid.y()->size(), grid.z()->size(), grid.u()->size()});
  auto _result = result.template mutable_unchecked<4>();

  // Captures the detected exceptions in the calculation function
  // (only the last exception captured is kept)
  auto except = std::exception_ptr(nullptr);

  auto worker = [&](const size_t start, const size_t end) {
    try {
      // Access to the shared pointer outside the loop to avoid data races
      const auto &x_axis = *grid.x();
      const auto &y_axis = *grid.y();
      const auto &z_axis = *grid.z();
      auto x_frame = std::vector<int64_t>(nx * 2 + 1);
      auto y_frame = std::vector<int64_t>(ny * 2 + 1);

      for (size_t iu = start; iu < end; ++iu) {
        for (int64_t ix = 0; ix < x_axis.size(); ++ix) {
          auto x = x_axis(ix);

          // We retrieve the indexes framing the current value.
          frame_index(ix, x_axis.size(), x_axis.is_angle(), x_frame);

          // Read the first value of the calculated window.
          const auto x0 = x_axis(x_frame[0]);

          // The current value is normalized to the first value in the
          // window.
          if (x_axis.is_angle()) {
            x = detail::math::normalize_angle(x, x0, 360.0);
          }

          for (int64_t iy = 0; iy < y_axis.size(); ++iy) {
            for (int64_t iz = 0; iz < z_axis.size(); ++iz) {
              _result(ix, iy, iz, iu) = loess_<Type, Grid4D<Type, AxisType>>(
                  grid, nx, ny, value_type, x_axis, y_axis, x_frame, y_frame,
                  x0, x, ix, iy, iz, iu);
            }
          }
        }
      }
    } catch (...) {
      except = std::current_exception();
    }
  };

  {
    pybind11::gil_scoped_release release;
    detail::dispatch(worker, grid.u()->size(), num_threads);
  }
  return result;
}

}  // namespace pyinterp::fill
