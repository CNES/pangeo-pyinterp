# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for pyinterp.core.geohash module."""

from __future__ import annotations

import numpy as np
import pytest

from ...core import geohash


TEST_CASES: list[tuple[str, float, float]] = [
    ("77mkh2hcj7mz", -26.015434642, -26.173663656),
    ("wthnssq3w00x", 29.291182895, 118.331595326),
    ("z3jsmt1sde4r", 51.400326027, 154.228244707),
    ("18ecpnqdg4s1", -86.976900779, -106.90988479),
    ("u90suzhjqv2s", 51.49934315, 23.417648894),
    ("k940p3ewmmyq", -39.365655496, 25.636144008),
    ("6g4wv2sze6ms", -26.934429639, -52.496991862),
    ("jhfyx4dqnczq", -62.123898484, 49.178194037),
    ("j80g4mkqz3z9", -89.442648795, 68.659722351),
    ("hq9z7cjwrcw4", -52.156511416, 13.88362641),
]


def test_string_numpy() -> None:
    """Test geohash encoding and decoding with numpy arrays of strings."""
    # Test successful decoding with byte strings
    geohashes = np.array([item[0] for item in TEST_CASES], dtype="S")
    lons, lats = geohash.decode(geohashes, round=True)

    expected_lons = np.array([item[2] for item in TEST_CASES])
    expected_lats = np.array([item[1] for item in TEST_CASES])

    assert np.all(np.abs(lons - expected_lons) < 1e-6)
    assert np.all(np.abs(lats - expected_lats) < 1e-6)

    # Test error cases for decode
    with pytest.raises(ValueError):
        geohash.decode(
            np.array([item[0] for item in TEST_CASES], dtype="U"), round=True
        )

    with pytest.raises(ValueError):
        geohash.decode(
            geohashes.reshape(5, 2),  # type: ignore[arg-type]
            round=True,
        )

    with pytest.raises(ValueError):
        geohash.decode(np.array([b"0" * 24], dtype="S"), round=True)

    # Test where() with valid input
    stacked_geohashes = np.vstack((geohashes[:5], geohashes[5:]))
    indexes = geohash.where(stacked_geohashes)
    assert isinstance(indexes, dict)

    # Test error cases for where()
    with pytest.raises(ValueError):
        geohash.where(stacked_geohashes.astype("U"))

    with pytest.raises(ValueError):
        geohash.where(
            geohashes.reshape(  # type: ignore[arg-type]
                1,
                2,
                5,
            ),
        )


def test_where_keys() -> None:
    """Test that where() returns the full geohash codes as keys.

    Regression test for https://github.com/CNES/pangeo-pyinterp/issues/40
    """
    # The number of columns differs from the length of the codes.
    codes = np.array([[b"xyz", b"xyz"], [b"abc", b"abc"]], dtype="S3")
    assert geohash.where(codes) == {
        b"xyz": ((0, 0), (0, 1)),
        b"abc": ((1, 1), (0, 1)),
    }

    # More columns than the maximum length of a code.
    codes = np.full((2, 20), b"s", dtype="S1")
    assert geohash.where(codes) == {b"s": ((0, 1), (0, 19))}

    # Codes generated from a regular grid.
    lon, lat = np.meshgrid(
        np.arange(-180.0, 180.0, 5.0) + 2.5, np.arange(-90.0, 90.0, 5.0) + 2.5
    )
    codes = geohash.encode(lon.ravel(), lat.ravel(), precision=2)
    codes = codes.reshape(lon.shape)
    indexes = geohash.where(codes)
    assert set(indexes) == set(np.unique(codes).tolist())
    for code, ((row_min, row_max), (col_min, col_max)) in indexes.items():
        rows, cols = np.nonzero(codes == code)
        assert (row_min, row_max) == (rows.min(), rows.max())
        assert (col_min, col_max) == (cols.min(), cols.max())
        assert np.all(
            codes[row_min : row_max + 1, col_min : col_max + 1] == code
        )


def test_where_bounds() -> None:
    """Test that where() bounds cover all the occurrences of a code.

    Regression test for https://github.com/CNES/pangeo-pyinterp/issues/40
    """
    # Two occurrences only
    codes = np.array([[b"s"], [b"s"]], dtype="S1")
    assert geohash.where(codes) == {b"s": ((0, 1), (0, 0))}
    assert geohash.where(codes.T) == {b"s": ((0, 0), (0, 1))}

    # Arrays that are not C-contiguous
    codes = np.array(
        [[b"s0", b"s0", b"e1"], [b"e1", b"e1", b"e1"]], dtype="S2"
    )
    expected = {b"s0": ((0, 1), (0, 0)), b"e1": ((0, 2), (0, 1))}
    assert geohash.where(codes.T) == expected
    assert geohash.where(np.asfortranarray(codes.T)) == expected
    assert geohash.where(codes[:, ::2]) == {
        b"s0": ((0, 0), (0, 0)),
        b"e1": ((0, 1), (0, 1)),
    }

    # Disconnected occurrences
    codes = np.array([[b"s"], [b"e"], [b"s"]], dtype="S1")
    assert geohash.where(codes) == {
        b"s": ((0, 2), (0, 0)),
        b"e": ((1, 1), (0, 0)),
    }

    codes = np.array(
        [[b"s0", b"e1", b"e1"], [b"e1", b"e1", b"e1"], [b"e1", b"e1", b"s0"]],
        dtype="S2",
    )
    assert geohash.where(codes) == {
        b"s0": ((0, 2), (0, 2)),
        b"e1": ((0, 2), (0, 2)),
    }


def test_bounding_zoom() -> None:
    """Test the transform function."""
    bboxes = geohash.bounding_boxes(precision=1)
    assert len(bboxes) == 32

    zoom_in = geohash.transform(bboxes, precision=3)
    assert len(zoom_in) == 2**10 * 32
    assert np.all(
        np.sort(geohash.transform(zoom_in, precision=1)) == np.sort(bboxes)
    )
