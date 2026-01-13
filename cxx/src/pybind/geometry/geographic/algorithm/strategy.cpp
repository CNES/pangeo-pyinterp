// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"

#include <nanobind/nanobind.h>

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kStrategyDoc = R"doc(
Geodetic calculation strategy.

Available strategies:
    - ANDOYER: Andoyer method - fast but less accurate
    - KARNEY: Karney method - most accurate but slower
    - THOMAS: Thomas method - balanced accuracy and performance
    - VINCENTY: Vincenty method - good balance (default)
)doc";

auto init_strategy(nanobind::module_& m) -> void {
  // Strategy enum
  nanobind::enum_<StrategyMethod>(m, "Strategy", kStrategyDoc)
      .value("ANDOYER", StrategyMethod::kAndoyer, "Andoyer method")
      .value("KARNEY", StrategyMethod::kKarney, "Karney method")
      .value("THOMAS", StrategyMethod::kThomas, "Thomas method")
      .value("VINCENTY", StrategyMethod::kVincenty, "Vincenty method (default)")
      .export_values();
}

}  // namespace pyinterp::geometry::geographic::pybind
