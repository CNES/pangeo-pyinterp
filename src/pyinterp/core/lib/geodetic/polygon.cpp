#include "pyinterp/geodetic/polygon.hpp"

#include "pyinterp/geodetic/box.hpp"

namespace pyinterp::geodetic {

/// Calculates the envelope of this polygon.
[[nodiscard]] auto Polygon::envelope() const -> Box {
  auto box = Box();
  boost::geometry::envelope(*this, box);
  return box;
}

}  // namespace pyinterp::geodetic