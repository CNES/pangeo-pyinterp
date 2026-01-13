# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Polygon class."""

import pickle

import numpy as np

from .....core.geometry.geographic import Polygon, Ring


def make_square_ring(
    x0: float = 0.0, y0: float = 0.0, size: float = 10.0
) -> Ring:
    """Create a square ring."""
    lon = np.array([x0, x0 + size, x0 + size, x0, x0], dtype=float)
    lat = np.array([y0, y0, y0 + size, y0 + size, y0], dtype=float)
    return Ring(lon, lat)


def test_polygon_creation_and_len() -> None:
    """Test Polygon creation and length properties."""
    outer = make_square_ring()
    polygon = Polygon(outer)
    assert bool(polygon) is True
    assert len(polygon.outer) == 5
    assert len(polygon.inners) == 0
    # Add two holes
    polygon.append(make_square_ring(2.0, 2.0, 3.0))
    polygon.append(make_square_ring(5.0, 5.0, 2.0))
    assert len(polygon.outer) == 5
    assert len(polygon.inners) == 2
    assert len(polygon.inners[0]) == 5
    assert len(polygon.inners[1]) == 5


def test_inners_proxy_rw_and_iteration() -> None:
    """Test Polygon inners proxy read/write and iteration."""
    outer = make_square_ring()
    hole1 = make_square_ring(2.0, 2.0, 3.0)
    hole2 = make_square_ring(5.0, 5.0, 2.0)

    polygon = Polygon(outer, [hole1])

    # Read view
    view = polygon.inners
    assert len(view) == 1
    assert isinstance(view[0], Ring)

    # Set item through view
    view[0] = hole2
    assert len(polygon.inners) == 1
    assert str(polygon).startswith("Polygon[outer=")

    # Append via view
    view.append(hole1)
    assert len(polygon.inners) == 2

    # Iterate
    iterated = list(view)
    assert isinstance(iterated[0], Ring)
    assert isinstance(iterated[1], Ring)

    # Replace via property setter
    polygon.inners = [hole2]
    assert len(polygon.inners) == 1

    # Clear via view
    view = polygon.inners
    view.clear()
    assert len(polygon.inners) == 0


def test_pickle_roundtrip() -> None:
    """Test Polygon pickling and unpickling."""
    outer = make_square_ring()
    hole = make_square_ring(3.0, 3.0, 2.0)
    polygon = Polygon(outer, [hole])

    data = pickle.dumps(polygon)
    unpickled_polygon = pickle.loads(data)

    assert unpickled_polygon == polygon


def test_repr_and_str() -> None:
    """Test Polygon __repr__ and __str__ methods."""
    polygon = Polygon(make_square_ring())
    polygon_repr = repr(polygon)
    polygon_str = str(polygon)
    assert "Polygon(" in polygon_repr
    assert "Polygon[outer=" in polygon_str
