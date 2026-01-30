# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test MultiPolygon class."""

from __future__ import annotations

import pickle

import numpy as np

from .....core.geometry.geographic import MultiPolygon, Polygon, Ring


def make_square_ring(
    x0: float = 0.0, y0: float = 0.0, size: float = 10.0
) -> Ring:
    """Create a square ring."""
    lon = np.array([x0, x0 + size, x0 + size, x0, x0], dtype=float)
    lat = np.array([y0, y0, y0 + size, y0 + size, y0], dtype=float)
    return Ring(lon, lat)


def make_polygon_with_hole() -> Polygon:
    """Create a polygon with a hole."""
    outer = make_square_ring()
    hole = make_square_ring(2.0, 2.0, 3.0)
    return Polygon(outer, [hole])


def test_multipolygon_creation_and_len() -> None:
    """Test MultiPolygon creation and length."""
    multi_polygon = MultiPolygon()
    assert bool(multi_polygon) is False
    assert len(multi_polygon) == 0

    polygon_with_hole = make_polygon_with_hole()
    square_polygon = Polygon(make_square_ring(20.0, 0.0, 5.0))

    multi_polygon.append(polygon_with_hole)
    multi_polygon.append(square_polygon)

    assert bool(multi_polygon) is True
    assert len(multi_polygon) == 2


def test_multipolygon_indexing_and_view() -> None:
    """Test MultiPolygon indexing and polygons view."""
    polygon_with_hole = make_polygon_with_hole()
    square_polygon = Polygon(make_square_ring(20.0, 0.0, 5.0))
    multi_polygon = MultiPolygon([polygon_with_hole])

    # __getitem__/__setitem__
    assert isinstance(multi_polygon[0], Polygon)
    multi_polygon[0] = square_polygon
    assert isinstance(multi_polygon[0], Polygon)

    # View over polygons property (RW)
    view = multi_polygon.polygons
    assert len(view) == len(multi_polygon)
    view.append(polygon_with_hole)
    assert len(view) == 2
    view[0] = polygon_with_hole
    assert isinstance(view[0], Polygon)

    # Iterate
    items = list(view)
    assert len(items) == 2
    assert all(isinstance(x, Polygon) for x in items)

    # Replace via property setter
    multi_polygon.polygons = [square_polygon]
    assert len(multi_polygon) == 1


def test_multipolygon_pickle_roundtrip() -> None:
    """Test MultiPolygon pickling and unpickling."""
    polygon_with_hole = make_polygon_with_hole()
    square_polygon = Polygon(make_square_ring(20.0, 0.0, 5.0))
    multi_polygon = MultiPolygon([polygon_with_hole, square_polygon])

    data = pickle.dumps(multi_polygon)
    mq = pickle.loads(data)

    assert len(mq) == len(multi_polygon)
    assert bool(mq) is True


def test_multipolygon_repr_str() -> None:
    """Test MultiPolygon __repr__ and __str__ methods."""
    multi_polygon = MultiPolygon([make_polygon_with_hole()])
    multi_polygon_repr = repr(multi_polygon)
    multi_polygon_str = str(multi_polygon)
    assert "MultiPolygon(" in multi_polygon_repr
    assert "MultiPolygon[n=" in multi_polygon_str
