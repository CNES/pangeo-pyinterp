import pytest
import pyinterp.geohash as geohash
import pyinterp.geodetic as geodetic


def test_index():
    # Create dummy data and populate the index
    data = dict(
        (key, key) for key in geohash.string.bounding_boxes(precision=3))
    store = geohash.storage.UnQlite(":mem:", mode="w")
    idx = geohash.index.init_geohash(store)
    idx.update(data)

    # index.box()
    box = geodetic.Box(geodetic.Point(-40, -40), geodetic.Point(40, 40))
    boxes = list(geohash.string.bounding_boxes(box, precision=3))
    assert idx.box(box) == boxes

    with pytest.raises(RuntimeError):
        idx = geohash.index.init_geohash(store)

    idx = geohash.index.open_geohash(store)
    assert idx.precision == 3
