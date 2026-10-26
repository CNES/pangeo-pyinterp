# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""GeoHash encoding, decoding and spatial indexing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyinterp.core import geohash
from pyinterp.core.geometry import geographic

from ._runner import Case, Context


if TYPE_CHECKING:
    from collections.abc import Callable


def _codes(ctx: Context, n: int, precision: int) -> np.ndarray:
    """Codes of random points."""
    rng = ctx.rng()
    return geohash.encode(
        rng.uniform(-180, 180, n), rng.uniform(-90, 90, n), precision
    )


def _encode(
    ctx: Context, n: int, precision: int
) -> Callable[[], Callable[[], object]]:
    def setup() -> Callable[[], object]:
        rng = ctx.rng()
        lon = rng.uniform(-180, 180, n)
        lat = rng.uniform(-90, 90, n)
        return lambda: geohash.encode(lon, lat, precision)

    return setup


def _on_codes(
    ctx: Context, n: int, func: Callable[[np.ndarray], object]
) -> Callable[[], Callable[[], object]]:
    def setup() -> Callable[[], object]:
        codes = _codes(ctx, n, 12)
        return lambda: func(codes)

    return setup


def _where(ctx: Context, shape: tuple[int, int]) -> Callable[[], object]:
    """Time the search of the cells sharing a code in a 2D grid of codes."""
    lon = np.linspace(-180, 180, shape[0])
    lat = np.linspace(-80, 80, shape[1])
    mx, my = np.meshgrid(lon, lat, indexing="ij")
    codes = geohash.encode(mx.ravel(), my.ravel(), 3).reshape(shape)
    return lambda: geohash.where(codes)


def _bounding_boxes_polygon(
    ctx: Context, precision: int
) -> Callable[[], Callable[[], object]]:
    """Time the enumeration of the cells covering a polygon."""

    def setup() -> Callable[[], object]:
        theta = np.linspace(0, 2 * np.pi, 360)
        ring = geographic.Ring(
            10 * np.cos(theta[::-1]), 10 * np.sin(theta[::-1])
        )
        polygon = geographic.Polygon(ring)
        return lambda: geohash.bounding_boxes(
            polygon, precision, num_threads=ctx.threads
        )

    return setup


def cases(ctx: Context) -> list[Case]:
    """Build the cases of the family."""
    n = ctx.size(2_000_000, 200_000)
    shape = ctx.size((1440, 720), (360, 180))
    precision = ctx.size(4, 3)
    box = geographic.Box((-20, -20), (20, 20))
    return [
        *(
            Case(f"encode/precision{p}", _encode(ctx, n, p), n)
            for p in (6, 12)
        ),
        Case("decode", _on_codes(ctx, n, geohash.decode), n),
        Case(
            "decode/round",
            _on_codes(ctx, n, lambda c: geohash.decode(c, round=True)),
            n,
        ),
        Case("area", _on_codes(ctx, n, geohash.area), n),
        Case(
            "transform/coarser",
            _on_codes(ctx, n, lambda c: geohash.transform(c, 6)),
            n,
        ),
        Case(
            "where",
            lambda: _where(ctx, shape),
            shape[0] * shape[1],
        ),
        Case(
            "bounding_boxes/box",
            lambda: lambda: geohash.bounding_boxes(box, precision),
            1,
        ),
        Case(
            "bounding_boxes/polygon",
            _bounding_boxes_polygon(ctx, precision),
            1,
        ),
    ]
