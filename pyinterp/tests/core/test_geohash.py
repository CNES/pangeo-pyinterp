# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for pyinterp.core.geohash module."""

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


def test_bounding_zoom() -> None:
    """Test the transform function."""
    bboxes = geohash.bounding_boxes(precision=1)
    assert len(bboxes) == 32

    zoom_in = geohash.transform(bboxes, precision=3)
    assert len(zoom_in) == 2**10 * 32
    assert np.all(
        np.sort(geohash.transform(zoom_in, precision=1)) == np.sort(bboxes)
    )
