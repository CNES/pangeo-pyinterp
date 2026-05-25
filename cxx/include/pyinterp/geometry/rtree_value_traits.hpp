// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::geometry {

/// @brief Traits describing how a value type is stored inside an RTree.
///
/// Specializations control three orthogonal aspects:
///
/// 1. **Promotion type** (`promotion_type`): the arithmetic type used when an
///    R-tree value is combined with a coordinate (e.g. inside IDW or
///    kriging). Defaults to @c Type itself, which is only meaningful for
///    arithmetic @c Type. Non-arithmetic values (such as
///    @ref pyinterp::math::interpolate::Observation) override this if needed.
///
/// 2. **Serialization** (@ref write / @ref read): how the trailing value of
///    each `(point, value)` record is encoded inside the
///    @ref pyinterp::serialization::Writer / @ref ...::Reader buffer used by
///    @c RTree::pack / @c RTree::unpack.
///
/// 3. **Serialization tag** (@ref serialization_tag): a small non-zero
///    contribution mixed into the R-tree magic number. The default is `0` so
///    existing pickles using scalar @c Type remain bit-compatible. Custom
///    value types should pick a unique tag to forbid cross-loading (e.g. a
///    pickle of `RTree<Point, Observation<double>>` must fail to load as
///    `RTree<Point, double>`).
template <typename Type>
struct value_traits {
  /// Underlying scalar type of the stored value.
  using scalar_type = Type;

  /// Type produced when combining a coordinate with the stored value.
  ///
  /// Conditional definition lets non-arithmetic @c Type instantiate the
  /// container without immediately requiring `coordinate + Type` to compile.
  /// Math methods that actually need promotion are constrained by
  /// @c requires std::floating_point<Type> at the member level.
  template <typename Coordinate>
  using promotion_type = std::conditional_t<
      std::is_arithmetic_v<Type>,
      decltype(std::declval<Coordinate>() + std::declval<Type>()), Type>;

  /// Distinguishing tag mixed into the RTree magic number. `0` for backward
  /// compatibility with the original scalar layout; non-zero for value types
  /// that change the on-disk record size.
  static constexpr uint32_t serialization_tag = 0;

  /// Write a value into the serialization buffer.
  static auto write(serialization::Writer& buffer, const Type& value) -> void {
    buffer.write(value);
  }

  /// Read a value from the serialization buffer.
  [[nodiscard]] static auto read(serialization::Reader& state) -> Type {
    return state.read<Type>();
  }
};

}  // namespace pyinterp::geometry
