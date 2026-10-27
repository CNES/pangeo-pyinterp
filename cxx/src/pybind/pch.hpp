// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// Precompiled header of the pyinterp.core extension (PYINTERP_USE_PCH).
//
// Parsing these third-party headers accounts for about half of the CPU time
// spent compiling the extension, <boost/geometry.hpp> alone costing almost two
// seconds per translation unit. Only stable third-party and standard headers
// belong here: a project header would invalidate the precompiled header, and
// so rebuild every translation unit, each time it is modified.
#pragma once

// IWYU pragma: begin_keep
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Core>
#include <algorithm>
#include <boost/geometry.hpp>
#include <cstdint>
#include <format>
#include <optional>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
// IWYU pragma: end_keep
