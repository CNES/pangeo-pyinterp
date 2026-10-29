// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/reduced_gaussian.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/config/reduced_gaussian.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/reduced_gaussian/grid.hpp"
#include "pyinterp/geometry/reduced_gaussian/interpolate.hpp"
#include "pyinterp/geometry/reduced_gaussian/mesh.hpp"
#include "pyinterp/geometry/reduced_gaussian/zone.hpp"
#include "pyinterp/parallel_for.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::reduced_gaussian::pybind {
namespace {

using Config = config::reduced_gaussian::ReducedGaussian;
using Blend = config::reduced_gaussian::Blend;
using Ranges = Eigen::Matrix<int64_t, Eigen::Dynamic, 2, Eigen::RowMajor>;
using CornerIndex = Eigen::Matrix<int64_t, Eigen::Dynamic, 4, Eigen::RowMajor>;
using CornerWeight = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;
using RowMajorMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// @brief Signature of a native column kernel (a numba `cfunc`, a C
///        function). Returns 0 on success.
using NativeKernel = int (*)(const double* columns, int64_t n_fields,
                             const int64_t* widths, const double* a,
                             const double* b, int64_t n_half_levels,
                             const double* point, int64_t n_point, double* out,
                             int64_t n_out);

/// @brief Tag of the calling convention of `NativeKernel`, which a kernel
///        object states in its `abi` attribute.
constexpr const char* const kKernelAbi =
    "pyinterp.reduced_gaussian.column_kernel/1";

/// @brief A grid that keeps the arrays of its fields alive.
class PyGrid : public Grid {
 public:
  PyGrid(std::shared_ptr<const Mesh> mesh, std::shared_ptr<const Zone> zone,
         std::vector<Field> fields, Vector<double> a, Vector<double> b,
         std::vector<nb::object> owners)
      : Grid(std::move(mesh), std::move(zone), std::move(fields), std::move(a),
             std::move(b)),
        owners_(std::move(owners)) {}

  /// @brief The arrays of the fields, in order.
  [[nodiscard]] auto owners() const noexcept -> const std::vector<nb::object>& {
    return owners_;
  }

 private:
  std::vector<nb::object> owners_;
};

/// @brief Read the fields of a grid from a dict of arrays, without copying.
auto make_grid(const std::shared_ptr<const Mesh>& mesh, const nb::dict& fields,
               const std::shared_ptr<const Zone>& zone,
               const std::optional<Vector<double>>& a,
               const std::optional<Vector<double>>& b)
    -> std::shared_ptr<PyGrid> {
  std::vector<Field> views;
  std::vector<nb::object> owners;
  for (const auto& [key, value] : fields) {
    auto name = nb::cast<std::string>(nb::str(key));
    Field field;
    field.name = name;
    nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> array;
    if (!nb::try_cast(value, array, false)) {
      throw nb::type_error(
          ("field '" + name +
           "' must be a C-contiguous numpy array (use numpy.ascontiguousarray)")
              .c_str());
    }
    if (array.dtype() == nb::dtype<float>()) {
      field.dtype = DType::kFloat32;
    } else if (array.dtype() == nb::dtype<double>()) {
      field.dtype = DType::kFloat64;
    } else {
      throw nb::type_error(
          ("field '" + name + "' must hold float32 or float64 values").c_str());
    }
    if (array.ndim() == 1) {
      field.width = 1;
    } else if (array.ndim() == 2) {
      field.width = static_cast<int64_t>(array.shape(1));
      field.has_levels = true;
    } else {
      throw std::invalid_argument("field '" + name +
                                  "' must have shape (points,) or (points, "
                                  "levels)");
    }
    field.rows = static_cast<int64_t>(array.shape(0));
    field.data = array.data();
    views.push_back(std::move(field));
    owners.push_back(nb::borrow(value));
  }
  return std::make_shared<PyGrid>(
      mesh, zone, std::move(views), a.value_or(Vector<double>{}),
      b.value_or(Vector<double>{}), std::move(owners));
}

/// @brief A grid or a series, as given to `interpolate` and `apply`.
struct Source {
  /// @brief The series; a single grid is a series of one step.
  std::shared_ptr<Series> series;
  /// @brief Whether the source is a series, which reads the dates.
  bool is_series{false};
};

/// @brief Read the source argument.
auto to_source(const nb::handle& source) -> Source {
  std::shared_ptr<Series> series;
  if (nb::try_cast(source, series, false)) {
    return {.series = std::move(series), .is_series = true};
  }
  std::shared_ptr<PyGrid> grid;
  if (nb::try_cast(source, grid, false)) {
    auto single = std::make_shared<Series>(grid->mesh_ptr());
    single->append(0, std::move(grid));
    return {.series = std::move(single), .is_series = false};
  }
  throw nb::type_error("source must be a Grid or a Series");
}

/// @brief Indices of the fields named, or of every field.
auto select_fields(const Series& series,
                   const std::optional<std::vector<std::string>>& names)
    -> std::vector<size_t> {
  if (series.size() == 0) {
    throw std::invalid_argument("the series has no step");
  }
  const auto& fields = series.grid(0).fields();
  std::vector<size_t> result;
  if (!names) {
    for (size_t ix = 0; ix < fields.size(); ++ix) {
      result.push_back(ix);
    }
    return result;
  }
  for (const auto& name : *names) {
    const auto it = std::ranges::find(fields, name, &Field::name);
    if (it == fields.end()) {
      throw std::invalid_argument("the grid has no field '" + name + "'");
    }
    result.push_back(static_cast<size_t>(it - fields.begin()));
  }
  return result;
}

/// @brief The rule that chooses the steps of a date.
auto bracket_rule(const Config& config) -> BracketRule {
  return {.nearest = config.time_method() == config::AxisMethod::kNearest,
          .interval = config.interval(),
          .max_step_gap = config.max_step_gap()};
}

/// @brief Check the dates against the source and the points.
auto check_times(const Source& source,
                 const std::optional<ConstVectorRef<int64_t>>& time, int64_t n)
    -> const int64_t* {
  if (!source.is_series) {
    if (time) {
      throw std::invalid_argument(
          "dates are given for a Grid, which has a single step: use a "
          "Series");
    }
    return nullptr;
  }
  if (!time) {
    throw std::invalid_argument("a Series needs the dates of the points");
  }
  if (time->size() != n) {
    throw std::invalid_argument("time must have the same size as lon and lat");
  }
  return time->data();
}

/// @brief A float64 numpy array that owns its buffer, NaN-filled.
auto make_array(int64_t rows, int64_t width, bool has_levels)
    -> std::pair<nb::object, double*> {
  auto* data = new double[static_cast<size_t>(rows * width)];
  std::fill_n(data, rows * width, std::numeric_limits<double>::quiet_NaN());
  nb::capsule owner(
      data, [](void* ptr) noexcept { delete[] static_cast<double*>(ptr); });
  if (has_levels) {
    return {nb::ndarray<nb::numpy, double, nb::ndim<2>>(
                data, {static_cast<size_t>(rows), static_cast<size_t>(width)},
                owner)
                .cast(),
            data};
  }
  return {nb::ndarray<nb::numpy, double, nb::ndim<1>>(
              data, {static_cast<size_t>(rows)}, owner)
              .cast(),
          data};
}

/// @brief One float64 array per field, to receive the columns of points.
struct Columns {
  nb::dict arrays;
  std::vector<double*> data;

  Columns(const Evaluator& evaluator, int64_t rows) {
    for (size_t ix = 0; ix < evaluator.num_fields(); ++ix) {
      const auto& field = evaluator.field(ix);
      auto [array, ptr] = make_array(rows, field.width, field.has_levels);
      arrays[field.name.c_str()] = array;
      data.push_back(ptr);
    }
  }

  /// @brief Scatter a concatenated column into row `row` of the arrays.
  void scatter(const Evaluator& evaluator, int64_t row,
               const double* column) const {
    const auto& offsets = evaluator.offsets();
    for (size_t ix = 0; ix < data.size(); ++ix) {
      const auto width = offsets[ix + 1] - offsets[ix];
      std::copy_n(column + offsets[ix], width, data[ix] + (row * width));
    }
  }
};

/// @brief Interpolate the fields at the points.
auto interpolate(const nb::handle& source_arg,
                 const ConstVectorRef<double>& lon,
                 const ConstVectorRef<double>& lat,
                 const std::optional<ConstVectorRef<int64_t>>& time,
                 const std::optional<std::vector<std::string>>& fields,
                 const Config& config) -> nb::dict {
  broadcast::check_eigen_shape("lon", lon, "lat", lat);
  const auto source = to_source(source_arg);
  const auto n = lon.size();
  const auto* times = check_times(source, time, n);
  const Evaluator evaluator(*source.series,
                            select_fields(*source.series, fields), times,
                            bracket_rule(config));
  Columns result(evaluator, n);
  const auto bounds_error = config.common().bounds_error();
  {
    nb::gil_scoped_release release;
    parallel_for(
        n,
        [&](int64_t start, int64_t end) {
          auto scratch = evaluator.make_scratch();
          std::vector<double> column(
              static_cast<size_t>(evaluator.total_width()));
          int64_t failed_step = 0;
          for (auto ix = start; ix < end; ++ix) {
            const auto status = evaluator.evaluate(
                ix, lon[ix], lat[ix], column.data(), scratch, failed_step);
            if (status == Status::kOk) {
              result.scatter(evaluator, ix, column.data());
            } else if (bounds_error) {
              evaluator.raise(status, ix, lon[ix], lat[ix], failed_step);
            }
          }
        },
        config.common().num_threads());
  }
  return result.arrays;
}

/// @brief The values attached to each point, besides its columns.
struct PointArgs {
  std::vector<std::string> names;
  std::vector<Vector<double>> values;
};

/// @brief Read the extra arguments of the points.
auto to_point_args(const std::optional<nb::dict>& args, int64_t n)
    -> PointArgs {
  PointArgs result;
  if (!args) {
    return result;
  }
  for (const auto& [key, value] : *args) {
    auto name = nb::cast<std::string>(nb::str(key));
    auto vector = nb::cast<Vector<double>>(value);
    if (vector.size() != n) {
      throw std::invalid_argument("argument '" + name +
                                  "' must have the same size as lon and lat");
    }
    result.names.push_back(std::move(name));
    result.values.push_back(std::move(vector));
  }
  return result;
}

/// @brief A numpy copy of a block of a vector.
template <typename T>
auto block_array(const T* data, int64_t start, int64_t count) -> nb::object {
  auto* copy = new T[static_cast<size_t>(count)];
  std::copy_n(data + start, count, copy);
  nb::capsule owner(copy,
                    [](void* ptr) noexcept { delete[] static_cast<T*>(ptr); });
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(
             copy, {static_cast<size_t>(count)}, owner)
      .cast();
}

/// @brief The hybrid coefficient of the series as a numpy array, or None.
auto coefficient(const Vector<double>& values) -> nb::object {
  if (values.size() == 0) {
    return nb::none();
  }
  return nb::cast(values);
}

/// @brief Read the result of a batched callback into `out`, (count, n_out).
void read_callback_result(const nb::object& result, int64_t count,
                          int64_t n_out, std::vector<double>& out) {
  auto values =
      nb::cast<RowMajorMatrixXd>(nb::module_::import_("numpy")
                                     .attr("asarray")(result, "float64")
                                     .attr("reshape")(count, -1));
  if (values.cols() != n_out) {
    throw std::invalid_argument(
        "the callback returned " + std::to_string(values.cols()) +
        " values per point, expected n_out=" + std::to_string(n_out));
  }
  out.assign(values.data(), values.data() + (count * n_out));
}

/// @brief Run a Python callable on blocks of points.
auto apply_python(const Source& source, const nb::callable& callback,
                  const ConstVectorRef<double>& lon,
                  const ConstVectorRef<double>& lat, const int64_t* times,
                  const PointArgs& args,
                  const std::vector<size_t>& field_indices, int64_t n_out,
                  const Config& config, double* output) -> void {
  const auto n = lon.size();
  const Evaluator evaluator(*source.series, field_indices, times,
                            bracket_rule(config));
  const auto results = config.blend() == Blend::kResults;
  const auto bounds_error = config.common().bounds_error();
  const auto& first = source.series->grid(0);
  const auto a = coefficient(first.a());
  const auto b = coefficient(first.b());
  const auto chunk = config.chunk_size();

  for (int64_t start = 0; start < n; start += chunk) {
    const auto count = std::min(chunk, n - start);
    Columns early(evaluator, count);
    std::optional<Columns> later;
    if (results) {
      later.emplace(evaluator, count);
    }
    std::vector<double> weight(static_cast<size_t>(count), 0.0);
    std::vector<uint8_t> valid(static_cast<size_t>(count), 0);
    {
      nb::gil_scoped_release release;
      parallel_for(
          count,
          [&](int64_t begin, int64_t end) {
            auto scratch = evaluator.make_scratch();
            const auto width = static_cast<size_t>(evaluator.total_width());
            std::vector<double> column(width);
            std::vector<double> second(width);
            for (auto jx = begin; jx < end; ++jx) {
              const auto ix = start + jx;
              int64_t failed_step = 0;
              Status status = Status::kOk;
              if (!results) {
                status = evaluator.evaluate(ix, lon[ix], lat[ix], column.data(),
                                            scratch, failed_step);
                if (status == Status::kOk) {
                  early.scatter(evaluator, jx, column.data());
                }
              } else {
                Bracket bracket;
                status = evaluator.evaluate_steps(
                    ix, lon[ix], lat[ix], column.data(), second.data(), bracket,
                    scratch, failed_step);
                if (status == Status::kOk) {
                  early.scatter(evaluator, jx, column.data());
                  // A single step is its own later step: the blend of the
                  // two results then gives that of the step.
                  later->scatter(
                      evaluator, jx,
                      bracket.single() ? column.data() : second.data());
                  weight[static_cast<size_t>(jx)] = bracket.weight;
                }
              }
              if (status == Status::kOk) {
                valid[static_cast<size_t>(jx)] = 1;
              } else if (bounds_error) {
                evaluator.raise(status, ix, lon[ix], lat[ix], failed_step);
              }
            }
          },
          config.common().num_threads());
    }

    nb::dict points;
    points["lon"] = block_array(lon.data(), start, count);
    points["lat"] = block_array(lat.data(), start, count);
    if (times != nullptr) {
      points["time"] = block_array(times, start, count);
    }
    for (size_t kx = 0; kx < args.names.size(); ++kx) {
      points[args.names[kx].c_str()] =
          block_array(args.values[kx].data(), start, count);
    }

    std::vector<double> r0;
    read_callback_result(callback(early.arrays, points, a, b), count, n_out,
                         r0);
    std::vector<double> r1;
    if (results) {
      read_callback_result(callback(later->arrays, points, a, b), count, n_out,
                           r1);
    }
    for (int64_t jx = 0; jx < count; ++jx) {
      if (valid[static_cast<size_t>(jx)] == 0) {
        continue;
      }
      const auto w = weight[static_cast<size_t>(jx)];
      for (int64_t kx = 0; kx < n_out; ++kx) {
        const auto pos = static_cast<size_t>((jx * n_out) + kx);
        output[((start + jx) * n_out) + kx] =
            results ? ((1.0 - w) * r0[pos]) + (w * r1[pos]) : r0[pos];
      }
    }
  }
}

/// @brief Run a native kernel on every point, in parallel and without the
///        GIL.
auto apply_native(const Source& source, NativeKernel kernel,
                  const ConstVectorRef<double>& lon,
                  const ConstVectorRef<double>& lat, const int64_t* times,
                  const PointArgs& args,
                  const std::vector<size_t>& field_indices, int64_t n_out,
                  const Config& config, double* output) -> void {
  const auto n = lon.size();
  const Evaluator evaluator(*source.series, field_indices, times,
                            bracket_rule(config));
  const auto results = config.blend() == Blend::kResults;
  const auto bounds_error = config.common().bounds_error();
  const auto& first = source.series->grid(0);
  const auto* a = first.a().size() != 0 ? first.a().data() : nullptr;
  const auto* b = first.b().size() != 0 ? first.b().data() : nullptr;
  const auto n_half = first.a().size();
  std::vector<int64_t> widths;
  widths.reserve(evaluator.num_fields());
  for (size_t ix = 0; ix < evaluator.num_fields(); ++ix) {
    widths.push_back(evaluator.field(ix).width);
  }
  const auto n_point =
      2 + (times != nullptr ? 1 : 0) + static_cast<int64_t>(args.names.size());
  const auto n_fields = static_cast<int64_t>(evaluator.num_fields());

  nb::gil_scoped_release release;
  parallel_for(
      n,
      [&](int64_t start, int64_t end) {
        auto scratch = evaluator.make_scratch();
        const auto width = static_cast<size_t>(evaluator.total_width());
        std::vector<double> early(width);
        std::vector<double> later(width);
        std::vector<double> point(static_cast<size_t>(n_point));
        std::vector<double> r0(static_cast<size_t>(n_out));
        std::vector<double> r1(static_cast<size_t>(n_out));
        for (auto ix = start; ix < end; ++ix) {
          point[0] = lon[ix];
          point[1] = lat[ix];
          size_t pos = 2;
          if (times != nullptr) {
            point[pos++] = static_cast<double>(times[ix]);
          }
          for (const auto& values : args.values) {
            point[pos++] = values[ix];
          }
          int64_t failed_step = 0;
          Status status = Status::kOk;
          int code = 0;
          double w = 0.0;
          bool two = false;
          if (!results) {
            status = evaluator.evaluate(ix, lon[ix], lat[ix], early.data(),
                                        scratch, failed_step);
          } else {
            Bracket bracket;
            status = evaluator.evaluate_steps(ix, lon[ix], lat[ix],
                                              early.data(), later.data(),
                                              bracket, scratch, failed_step);
            two = !bracket.single();
            w = bracket.weight;
          }
          if (status != Status::kOk) {
            if (bounds_error) {
              evaluator.raise(status, ix, lon[ix], lat[ix], failed_step);
            }
            continue;
          }
          code = kernel(early.data(), n_fields, widths.data(), a, b, n_half,
                        point.data(), n_point, r0.data(), n_out);
          if (code == 0 && two) {
            code = kernel(later.data(), n_fields, widths.data(), a, b, n_half,
                          point.data(), n_point, r1.data(), n_out);
          }
          if (code != 0) {
            if (bounds_error) {
              throw std::runtime_error("the kernel returned " +
                                       std::to_string(code) + " at point " +
                                       std::to_string(ix));
            }
            continue;
          }
          for (int64_t kx = 0; kx < n_out; ++kx) {
            const auto k = static_cast<size_t>(kx);
            output[(ix * n_out) + kx] =
                two ? ((1.0 - w) * r0[k]) + (w * r1[k]) : r0[k];
          }
        }
      },
      config.common().num_threads());
}

/// @brief Compute a value per point from the interpolated columns.
auto apply(const nb::handle& source_arg, const nb::object& callback,
           const ConstVectorRef<double>& lon, const ConstVectorRef<double>& lat,
           const std::optional<ConstVectorRef<int64_t>>& time,
           const std::optional<nb::dict>& args,
           const std::optional<std::vector<std::string>>& fields, int64_t n_out,
           const Config& config) -> nb::object {
  broadcast::check_eigen_shape("lon", lon, "lat", lat);
  if (n_out < 1) {
    throw std::invalid_argument("n_out must be positive");
  }
  const auto source = to_source(source_arg);
  const auto n = lon.size();
  const auto* times = check_times(source, time, n);
  const auto point_args = to_point_args(args, n);
  const auto field_indices = select_fields(*source.series, fields);
  auto [result, output] = make_array(n, n_out, n_out > 1);

  if (nb::hasattr(callback, "address")) {
    // Calling a function of another signature through this one would crash
    // the process: the kernel states the convention it follows.
    if (!nb::hasattr(callback, "abi") ||
        !nb::isinstance<nb::str>(callback.attr("abi")) ||
        nb::cast<std::string>(callback.attr("abi")) != kKernelAbi) {
      throw nb::type_error(("a native kernel must declare abi = '" +
                            std::string(kKernelAbi) +
                            "' (use pyinterp.reduced_gaussian.column_kernel)")
                               .c_str());
    }
    const auto address = nb::cast<uintptr_t>(callback.attr("address"));
    if (address == 0) {
      throw std::invalid_argument("the kernel address is null");
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    const auto kernel = reinterpret_cast<NativeKernel>(address);
    apply_native(source, kernel, lon, lat, times, point_args, field_indices,
                 n_out, config, output);
  } else if (PyCallable_Check(callback.ptr()) != 0) {
    apply_python(source, nb::borrow<nb::callable>(callback), lon, lat, times,
                 point_args, field_indices, n_out, config, output);
  } else {
    throw nb::type_error(
        "callback must be a callable or a native kernel with an 'address'");
  }
  return result;
}

constexpr const char* const kInterpolateDocstring = R"(
Interpolate the fields of a reduced Gaussian grid or series at points.

Args:
    source: A Grid, or a Series of grids in time.
    lon: Longitudes of the points, in degrees.
    lat: Latitudes of the points, in degrees.
    time: Dates of the points, as int64 in the unit of the series; required
        for a Series, refused for a Grid.
    fields: Names of the fields to interpolate; all by default.
    config: Configuration of the interpolation.

Returns:
    A dict from field name to its values: shape (n,) for a field given as
    (points,), (n, levels) for a field given as (points, levels).

Raises:
    IndexError: If a point is outside the mesh, has a date outside the
        series, or reads a corner a zone does not hold, and
        ``config.bounds_error`` is set.
)";

constexpr const char* const kApplyDocstring = R"(
Compute values per point from the interpolated columns of the fields.

Args:
    source: A Grid, or a Series of grids in time.
    callback: Either a Python callable, called on blocks of points as
        ``callback(columns, points, a, b)`` with ``columns`` a dict of the
        interpolated fields (as returned by :func:`interpolate`), ``points``
        a dict holding ``lon``, ``lat``, ``time`` (for a series) and the
        extra arguments, ``a`` and ``b`` the hybrid coefficients (or None);
        it returns an array of shape (block,) or (block, n_out). Or a native
        kernel: an object with an ``abi`` attribute equal to
        ``"pyinterp.reduced_gaussian.column_kernel/1"`` and an ``address``
        attribute pointing to a C function ``int kernel(const double* columns, int64 n_fields, const
        int64* widths, const double* a, const double* b, int64
        n_half_levels, const double* point, int64 n_point, double* out,
        int64 n_out)``, called once per point from several threads without
        the GIL; ``point`` holds lon, lat, time (for a series) and the extra
        arguments.
    lon: Longitudes of the points, in degrees.
    lat: Latitudes of the points, in degrees.
    time: Dates of the points, for a Series.
    args: Extra values per point, passed to the callback by name.
    fields: Names of the fields to interpolate; all by default.
    n_out: Number of values computed per point.
    config: Configuration of the interpolation.

Returns:
    The values, shape (n,) if ``n_out`` is 1, else (n, n_out). A point that
    cannot be interpolated, or for which the kernel returns a non-zero code,
    is NaN.
)";

}  // namespace

auto bind(nb::module_& m) -> void {
  nb::class_<Mesh>(m, "Mesh",
                   "A reduced Gaussian mesh: parallels of equally spaced "
                   "points, north to south, starting at longitude 0.")
      .def(nb::init<Vector<double>, Vector<int64_t>>(), nb::arg("latitudes"),
           nb::arg("points_per_parallel"),
           "Build the mesh from the latitudes of its parallels (strictly "
           "decreasing) and their number of points.")
      .def_prop_ro("latitudes", &Mesh::latitudes, "Latitudes of the parallels.")
      .def_prop_ro("points_per_parallel", &Mesh::points_per_parallel,
                   "Number of points of each parallel.")
      .def_prop_ro("offsets", &Mesh::offsets,
                   "Flat index of the first point of each parallel, then "
                   "the number of points.")
      .def_prop_ro("size", &Mesh::size, "Number of points.")
      .def_prop_ro("ny", &Mesh::ny, "Number of parallels.")
      .def(
          "locate",
          [](const Mesh& self, const ConstVectorRef<double>& lon,
             const ConstVectorRef<double>& lat)
              -> std::pair<CornerIndex, CornerWeight> {
            broadcast::check_eigen_shape("lon", lon, "lat", lat);
            CornerIndex index(lon.size(), 4);
            CornerWeight weight(lon.size(), 4);
            {
              nb::gil_scoped_release release;
              Cell cell;
              for (Eigen::Index ix = 0; ix < lon.size(); ++ix) {
                if (self.locate(lon[ix], lat[ix], cell)) {
                  for (Eigen::Index k = 0; k < 4; ++k) {
                    index(ix, k) = cell.index[static_cast<size_t>(k)];
                    weight(ix, k) = cell.weight[static_cast<size_t>(k)];
                  }
                } else {
                  index.row(ix).setConstant(-1);
                  weight.row(ix).setConstant(
                      std::numeric_limits<double>::quiet_NaN());
                }
              }
            }
            return {std::move(index), std::move(weight)};
          },
          "Locate the cells of points: the flat indices of their four "
          "corners (bottom right, bottom left, top right, top left; -1 "
          "outside the mesh) and their bilinear weights.",
          nb::arg("lon"), nb::arg("lat"))
      .def("__eq__",
           [](const Mesh& self, const Mesh& other) { return self == other; })
      .def("__getstate__",
           [](const Mesh& self) {
             return nb::make_tuple(self.latitudes(),
                                   self.points_per_parallel());
           })
      .def("__setstate__", [](Mesh& self, const nb::tuple& state) {
        if (state.size() != 2) {
          throw std::invalid_argument("invalid state");
        }
        new (&self) Mesh(nb::cast<Vector<double>>(state[0]),
                         nb::cast<Vector<int64_t>>(state[1]));
      });

  nb::class_<Zone>(m, "Zone",
                   "The points of a mesh a regional grid holds: sorted, "
                   "disjoint ranges [start, stop) of flat indices.")
      .def(
          "__init__",
          [](Zone* self, const Mesh& mesh,
             const Eigen::Ref<const Ranges>& ranges) {
            new (self) Zone(mesh.size(), ranges);
          },
          nb::arg("mesh"), nb::arg("ranges"),
          "Build a zone from ranges [start, stop) of flat indices, shape "
          "(k, 2).")
      .def_static("from_box", &Zone::from_box, nb::arg("mesh"),
                  nb::arg("lon_min"), nb::arg("lat_min"), nb::arg("lon_max"),
                  nb::arg("lat_max"),
                  "The zone a box needs: the bracketing parallels and, on "
                  "each, the bracketing points of every longitude.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static(
          "from_points",
          [](const Mesh& mesh, const ConstVectorRef<double>& lon,
             const ConstVectorRef<double>& lat, size_t num_threads) {
            if (num_threads == 0) {
              num_threads = std::thread::hardware_concurrency();
            }
            return Zone::from_points(mesh, lon, lat, num_threads);
          },
          nb::arg("mesh"), nb::arg("lon"), nb::arg("lat"),
          nb::arg("num_threads") = 0,
          "The zone holding exactly the corners the points read.",
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("size", &Zone::size, "Number of points held.")
      .def_prop_ro("mesh_size", &Zone::mesh_size,
                   "Number of points of the mesh.")
      .def_prop_ro("num_ranges", &Zone::num_ranges, "Number of ranges.")
      .def("ranges", &Zone::ranges,
           "The ranges [start, stop), shape (k, 2), sorted and disjoint.")
      .def("indices", &Zone::indices,
           "The flat indices of the points held, in storage order.")
      .def(
          "rows",
          [](const Zone& self, const ConstVectorRef<int64_t>& index) {
            Vector<int64_t> result(index.size());
            for (Eigen::Index ix = 0; ix < index.size(); ++ix) {
              result[ix] = self.row(index[ix]);
            }
            return result;
          },
          nb::arg("index"),
          "The rows of points of the mesh in the values, -1 if not held.")
      .def("__eq__",
           [](const Zone& self, const Zone& other) { return self == other; })
      .def("__getstate__",
           [](const Zone& self) {
             return nb::make_tuple(self.mesh_size(), self.ranges());
           })
      .def("__setstate__", [](Zone& self, const nb::tuple& state) {
        if (state.size() != 2) {
          throw std::invalid_argument("invalid state");
        }
        new (&self)
            Zone(nb::cast<int64_t>(state[0]), nb::cast<Ranges>(state[1]));
      });

  nb::class_<PyGrid>(m, "Grid",
                     "The fields of one time step on a mesh or a zone of it. "
                     "The arrays are read without copy.")
      .def(
          "__init__",
          [](PyGrid* self, const std::shared_ptr<Mesh>& mesh,
             const nb::dict& fields, const std::shared_ptr<Zone>& zone,
             const std::optional<Vector<double>>& a,
             const std::optional<Vector<double>>& b) {
            auto grid = make_grid(mesh, fields, zone, a, b);
            new (self) PyGrid(std::move(*grid));
          },
          nb::arg("mesh"), nb::arg("fields"), nb::arg("zone").none() = nullptr,
          nb::arg("a").none() = nb::none(), nb::arg("b").none() = nb::none(),
          "Build a grid from a dict of C-contiguous float32/float64 arrays of "
          "shape (points,) or (points, levels), with one row per point of the "
          "zone (or of the mesh without zone).")
      .def_prop_ro("mesh",
                   [](const PyGrid& self) {
                     return std::const_pointer_cast<Mesh>(self.mesh_ptr());
                   })
      .def_prop_ro("zone",
                   [](const PyGrid& self) {
                     return std::const_pointer_cast<Zone>(self.zone());
                   })
      .def_prop_ro("field_names",
                   [](const PyGrid& self) {
                     std::vector<std::string> names;
                     for (const auto& field : self.fields()) {
                       names.push_back(field.name);
                     }
                     return names;
                   })
      .def_prop_ro("fields",
                   [](const PyGrid& self) {
                     nb::dict result;
                     const auto& owners = self.owners();
                     for (size_t ix = 0; ix < owners.size(); ++ix) {
                       result[self.fields()[ix].name.c_str()] = owners[ix];
                     }
                     return result;
                   })
      .def_prop_ro("a",
                   [](const PyGrid& self) { return coefficient(self.a()); })
      .def_prop_ro("b",
                   [](const PyGrid& self) { return coefficient(self.b()); });

  nb::class_<Series>(m, "Series",
                     "A time series of grids on one mesh. Each step may hold "
                     "its own zone.")
      .def(
          "__init__",
          [](Series* self, const std::shared_ptr<Mesh>& mesh) {
            new (self) Series(mesh);
          },
          nb::arg("mesh"))
      .def(
          "append",
          [](Series& self, int64_t time, const std::shared_ptr<PyGrid>& grid) {
            self.append(time, grid);
          },
          nb::arg("time"), nb::arg("grid"),
          "Add a step at a date (int64, in the unit of the series).")
      .def("drop_before", &Series::drop_before, nb::arg("time"),
           "Remove the steps strictly before a date.")
      .def_prop_ro(
          "times",
          [](const Series& self) {
            const auto& times = self.times();
            return Vector<int64_t>(Eigen::Map<const Vector<int64_t>>(
                times.data(), static_cast<Eigen::Index>(times.size())));
          },
          "Dates of the steps, sorted.")
      .def_prop_ro(
          "mesh", [](const Series& self) -> const Mesh& { return self.mesh(); },
          nb::rv_policy::reference_internal)
      .def("__len__", &Series::size);

  m.def(
      "bracket",
      [](const ConstVectorRef<int64_t>& step_times,
         const ConstVectorRef<int64_t>& time, const Config& config)
          -> std::tuple<Vector<int64_t>, Vector<int64_t>, Vector<double>> {
        const std::vector<int64_t> times(step_times.begin(), step_times.end());
        for (size_t ix = 1; ix < times.size(); ++ix) {
          if (times[ix] <= times[ix - 1]) {
            throw std::invalid_argument(
                "step_times must be sorted and distinct");
          }
        }
        const auto rule = bracket_rule(config);
        Vector<int64_t> i0(time.size());
        Vector<int64_t> i1(time.size());
        Vector<double> weight(time.size());
        nb::gil_scoped_release release;
        for (Eigen::Index ix = 0; ix < time.size(); ++ix) {
          const auto found = Series::bracket(times, time[ix], rule);
          i0[ix] = found ? found->i0 : -1;
          i1[ix] = found ? found->i1 : -1;
          weight[ix] = found ? found->weight : 0.0;
        }
        return {std::move(i0), std::move(i1), std::move(weight)};
      },
      nb::arg("step_times"), nb::arg("time"), nb::arg("config"),
      "The steps each date reads, by the rule of a Series: (i0, i1, "
      "weight), with -1 for a date that reads no step and i1 == i0 for a "
      "date that reads a single one.");

  m.def("interpolate", &interpolate, nb::arg("source"), nb::arg("lon"),
        nb::arg("lat"), nb::arg("time").none() = nb::none(),
        nb::arg("fields").none() = nb::none(), nb::arg("config"),
        kInterpolateDocstring);

  m.def("apply", &apply, nb::arg("source"), nb::arg("callback"), nb::arg("lon"),
        nb::arg("lat"), nb::arg("time").none() = nb::none(),
        nb::arg("args").none() = nb::none(),
        nb::arg("fields").none() = nb::none(), nb::arg("n_out") = 1,
        nb::arg("config"), kApplyDocstring);
}

}  // namespace pyinterp::geometry::reduced_gaussian::pybind

namespace pyinterp::pybind {

auto init_reduced_gaussian(nb::module_& m) -> void {
  auto sub = m.def_submodule("reduced_gaussian",
                             "Interpolation of reduced Gaussian grids.");
  geometry::reduced_gaussian::pybind::bind(sub);
}

}  // namespace pyinterp::pybind
