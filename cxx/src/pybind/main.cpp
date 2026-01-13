// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/nanobind.h>

#include "pyinterp/pybind/axis.hpp"
#include "pyinterp/pybind/binning.hpp"
#include "pyinterp/pybind/config.hpp"
#include "pyinterp/pybind/dateutils.hpp"
#include "pyinterp/pybind/descriptive_statistics.hpp"
#include "pyinterp/pybind/fill.hpp"
#include "pyinterp/pybind/geohash.hpp"
#include "pyinterp/pybind/geometric.hpp"
#include "pyinterp/pybind/geometry.hpp"
#include "pyinterp/pybind/grid.hpp"
#include "pyinterp/pybind/histogram2d.hpp"
#include "pyinterp/pybind/period.hpp"
#include "pyinterp/pybind/rtree.hpp"
#include "pyinterp/pybind/tdigest.hpp"
#include "pyinterp/pybind/windowed.hpp"

NB_MODULE(core, m) {
  pyinterp::pybind::init_config(m);
  pyinterp::pybind::init_axis(m);
  pyinterp::pybind::init_grids(m);
  pyinterp::pybind::init_geometric(m);
  pyinterp::pybind::init_windowed(m);
  pyinterp::pybind::init_geometry(m);
  pyinterp::pybind::init_geohash(m);
  pyinterp::pybind::init_rtree_3d(m);
  pyinterp::pybind::init_binning(m);
  pyinterp::pybind::init_histogram2d(m);
  pyinterp::pybind::init_descriptive_statistics(m);
  pyinterp::pybind::init_tdigest(m);
  pyinterp::pybind::init_fill(m);
  pyinterp::pybind::init_dateutils(m);
  pyinterp::pybind::init_period(m);
}
