// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/nanobind.h>

#include <format>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

#include "pyinterp/pybind/grid.hpp"

namespace pyinterp::pybind {

namespace detail {

/// @brief Helper to determine the result type for interpolation.
/// Double, int64_t, or uint64_t produces double, everything else produces
/// float.
template <typename DataType>
using InterpolationResultType =
    std::conditional_t<std::is_same_v<DataType, double> ||
                           std::is_same_v<DataType, int64_t> ||
                           std::is_same_v<DataType, uint64_t>,
                       double, float>;

/// @brief Concept for checking if a grid is 1D.
template <typename GridType>
concept Is1DGrid = (GridType::kNDim == 1);

/// @brief Concept for checking if a grid is 2D.
template <typename GridType>
concept Is2DGrid = (GridType::kNDim == 2);

/// @brief Concept for checking if a grid is 3D.
template <typename GridType>
concept Is3DGrid = (GridType::kNDim == 3);

/// @brief Concept for checking if a grid is 4D.
template <typename GridType>
concept Is4DGrid = (GridType::kNDim == 4);

/// @brief Concept for checking if a grid is temporal.
template <typename GridType>
concept IsTemporalGrid = GridType::kHasTemporalAxis;

/// @brief Concept for checking if a grid is spatial (non-temporal).
template <typename GridType>
concept IsSpatialGrid = !GridType::kHasTemporalAxis;

}  // namespace detail

/// @brief Helper class for dispatching operations on GridHolder to concrete
/// grid types.
///
/// This class provides static methods to dispatch operations using std::visit
/// on the underlying GridVariant.
///
/// @tparam Point Point type template (e.g., geometry::SphericalPoint)
template <template <class> class Point>
class GridDispatcher {
 public:
  /// @brief Determine the result type based on grid dtype.
  /// @param dtype_str The dtype string from the grid.
  /// @return "float32" or "float64"
  [[nodiscard]] static constexpr auto result_dtype(std::string_view dtype_str)
      -> std::string_view {
    if (dtype_str == "float64" || dtype_str == "int64" ||
        dtype_str == "uint64") {
      return "float64";
    }
    // All other types (float32, integers) produce float32 results
    return "float32";
  }

  /// @brief Check if the grid dtype produces float64 results.
  [[nodiscard]] static constexpr auto is_float64_result(
      std::string_view dtype_str) -> bool {
    return dtype_str == "float64";
  }

  /// @brief Dispatch univariate interpolation to the appropriate concrete grid
  /// type.
  /// @tparam ConfigType Configuration type.
  /// @tparam InterpolationFunc Interpolation function template.
  /// @param grid The grid holder.
  /// @param x X coordinates.
  /// @param config Configuration object.
  /// @param func The interpolation function - should be callable as
  /// func.template operator()<DataType, ResultType>(grid, x, config)
  /// @return nanobind::object containing the result vector.
  template <typename ConfigType, typename InterpolationFunc>
  static auto dispatch_univariate(const GridHolder& grid,
                                  const Eigen::Ref<const Vector<double>>& x,
                                  const ConfigType& config,
                                  InterpolationFunc&& func)
      -> nanobind::object {
    if (grid.ndim() != 1) {
      throw std::invalid_argument(
          std::format("univariate requires 1D grid, got {}D", grid.ndim()));
    }

    return grid.visit([&](const auto& concrete_grid) -> nanobind::object {
      using GridType = std::decay_t<decltype(concrete_grid)>;

      if constexpr (detail::Is1DGrid<GridType>) {
        using DataType = typename GridType::data_type;
        using ResultType = detail::InterpolationResultType<DataType>;
        return nanobind::cast(func.template operator()<DataType, ResultType>(
            concrete_grid, x, config));
      }
      std::unreachable();
    });
  }

  /// @brief Dispatch bivariate interpolation to the appropriate concrete grid
  /// type.
  /// @tparam ConfigType Configuration type.
  /// @tparam InterpolationFunc Interpolation function template.
  /// @param grid The grid holder.
  /// @param x X coordinates.
  /// @param y Y coordinates.
  /// @param config Configuration object.
  /// @param func The interpolation function - should be callable as
  /// func.template operator()<DataType, ResultType>(grid, x, y, config)
  /// @return nanobind::object containing the result vector.
  template <typename ConfigType, typename InterpolationFunc>
  static auto dispatch_bivariate(const GridHolder& grid,
                                 const Eigen::Ref<const Vector<double>>& x,
                                 const Eigen::Ref<const Vector<double>>& y,
                                 const ConfigType& config,
                                 InterpolationFunc&& func) -> nanobind::object {
    if (grid.ndim() != 2) {
      throw std::invalid_argument(
          std::format("bivariate requires 2D grid, got {}D", grid.ndim()));
    }

    return grid.visit([&](const auto& concrete_grid) -> nanobind::object {
      using GridType = std::decay_t<decltype(concrete_grid)>;

      if constexpr (detail::Is2DGrid<GridType>) {
        using DataType = typename GridType::data_type;
        using ResultType = detail::InterpolationResultType<DataType>;
        return nanobind::cast(func.template operator()<DataType, ResultType>(
            concrete_grid, x, y, config));
      }
      std::unreachable();
    });
  }

  /// @brief Dispatch trivariate interpolation to the appropriate concrete grid
  /// type.
  /// @tparam ConfigType Configuration type.
  /// @tparam InterpolationFunc Interpolation function template.
  /// @param grid The grid holder.
  /// @param x X coordinates.
  /// @param y Y coordinates.
  /// @param z Z coordinates (nanobind::object for temporal/spatial
  /// flexibility).
  /// @param config Configuration object.
  /// @param func The interpolation function - should be callable as
  /// func.template operator()<DataType, ResultType, ZType>(
  ///     grid, x, y, z, config)
  /// @return nanobind::object containing the result vector.
  template <typename ConfigType, typename InterpolationFunc>
  static auto dispatch_trivariate(const GridHolder& grid,
                                  const Eigen::Ref<const Vector<double>>& x,
                                  const Eigen::Ref<const Vector<double>>& y,
                                  const nanobind::object& z_obj,
                                  const ConfigType& config,
                                  InterpolationFunc&& func)
      -> nanobind::object {
    if (grid.ndim() != 3) {
      throw std::invalid_argument(
          std::format("trivariate requires 3D grid, got {}D", grid.ndim()));
    }

    return grid.visit([&](const auto& concrete_grid) -> nanobind::object {
      using GridType = std::decay_t<decltype(concrete_grid)>;

      if constexpr (detail::Is3DGrid<GridType>) {
        using DataType = typename GridType::data_type;
        using ResultType = detail::InterpolationResultType<DataType>;

        if constexpr (detail::IsTemporalGrid<GridType>) {
          // Temporal grid: convert z to int64
          auto z = concrete_grid.template pybind_axis<2>().cast_to_int64(z_obj);
          return nanobind::cast(
              func.template operator()<DataType, ResultType, int64_t>(
                  concrete_grid, x, y, z, config));
        } else {
          // Spatial grid: z is double
          auto z = nanobind::cast<Eigen::Ref<const Vector<double>>>(z_obj);
          return nanobind::cast(
              func.template operator()<DataType, ResultType, double>(
                  concrete_grid, x, y, z, config));
        }
      }
      std::unreachable();
    });
  }

  /// @brief Dispatch quadrivariate interpolation to the appropriate concrete
  /// grid type.
  /// @tparam ConfigType Configuration type.
  /// @tparam InterpolationFunc Interpolation function template.
  /// @param grid The grid holder.
  /// @param x X coordinates.
  /// @param y Y coordinates.
  /// @param z Z coordinates (nanobind::object for temporal/spatial
  /// flexibility).
  /// @param u U coordinates.
  /// @param config Configuration object.
  /// @param func The interpolation function - should be callable as
  /// func.template operator()<DataType, ResultType, ZType>(
  ///     grid, x, y, z, u, config)
  /// @return nanobind::object containing the result vector.
  template <typename ConfigType, typename InterpolationFunc>
  static auto dispatch_quadrivariate(const GridHolder& grid,
                                     const Eigen::Ref<const Vector<double>>& x,
                                     const Eigen::Ref<const Vector<double>>& y,
                                     const nanobind::object& z_obj,
                                     const Eigen::Ref<const Vector<double>>& u,
                                     const ConfigType& config,
                                     InterpolationFunc&& func)
      -> nanobind::object {
    if (grid.ndim() != 4) {
      throw std::invalid_argument(
          std::format("quadrivariate requires 4D grid, got {}D", grid.ndim()));
    }

    return grid.visit([&](const auto& concrete_grid) -> nanobind::object {
      using GridType = std::decay_t<decltype(concrete_grid)>;

      if constexpr (detail::Is4DGrid<GridType>) {
        using DataType = typename GridType::data_type;
        using ResultType = detail::InterpolationResultType<DataType>;

        if constexpr (detail::IsTemporalGrid<GridType>) {
          // Temporal grid: convert z to int64
          auto z = concrete_grid.template pybind_axis<2>().cast_to_int64(z_obj);
          return nanobind::cast(
              func.template operator()<DataType, ResultType, int64_t>(
                  concrete_grid, x, y, z, u, config));
        } else {
          // Spatial grid: z is double
          auto z = nanobind::cast<Eigen::Ref<const Vector<double>>>(z_obj);
          return nanobind::cast(
              func.template operator()<DataType, ResultType, double>(
                  concrete_grid, x, y, z, u, config));
        }
      }
      std::unreachable();
    });
  }
};

}  // namespace pyinterp::pybind
