"""
Geogrophic Index
----------------
"""
from typing import Any, Dict, List, Optional
import json
import numpy
from . import lock
from . import storage
from .. import geodetic
from ..core.geohash import string


class GeoHash:
    """
    Geogrophic index based on GeoHash encoding.

    Args:
        store (MutableMapping): Object managing the storage of the index
        precision (int): Accuracy of the index. By default the precision is 3
            characters. The table below gives the correspondence between the
            number of characters (i.e. the ``precision`` parameter of this
            constructor), the size of the boxes of the grid at the equator and
            the total number of boxes.

            =========  ===============  ==========
            precision  lng/lat (km)     samples
            =========  ===============  ==========
            1          4950/4950        32
            2          618.75/1237.50   1024
            3          154.69/154.69    32768
            4          19.34/38.67      1048576
            5          4.83/4.83        33554432
            6          0.60/1.21        1073741824
            =========  ===============  ==========
        synchronizer (lock.Synchronizer, optional): Write synchronizer
    """
    def __init__(self,
                 store: storage.MutableMapping,
                 precision: int = 3,
                 synchronizer: Optional[lock.Synchronizer] = None) -> None:
        self._store = store
        self._precision = precision
        self._synchronizer = synchronizer or lock.PuppetSynchronizer()

    @property
    def store(self) -> storage.MutableMapping:
        """Gets the object hndling the storage of this instance"""
        return self._store

    @property
    def precision(self) -> int:
        """Accuracy of this instance"""
        return self._precision

    def set_properties(self) -> None:
        """Definition of index properties"""
        if b'.properties' in self._store:
            raise RuntimeError("index already initialized")
        self._store[b'.properties'] = json.dumps(
            {'precision': self._precision})

    @staticmethod
    def get_properties(store) -> Dict[str, Any]:
        """Reading index properties

        Return:
            dict: Index properties (number of character used to encode a
            position)
        """
        return json.loads(store[b'.properties'].pop())

    def encode(self, lon: numpy.ndarray, lat: numpy.ndarray) -> numpy.ndarray:
        """Encode points into geohash with the given precision

        Args:
            lon (numpy.ndarray): Longitudes in degrees of the positions to be
                encoded.
            lat (numpy.ndarray): Latitudes in degrees of the positions to be
                encoded.

        Return:
            numpy.ndarray: geohash code for each coordinates of the points
            read from the vectors provided.
        """
        return string.encode(lon, lat, precision=self._precision)

    def update(self, data: Dict[bytes, object]) -> None:
        """Update the index with the key/value pairs from data, overwriting
        existing keys.

        Args:
            data (dict): Geohash codes associated with the values to be stored
                in the database.
        """
        with self._synchronizer:
            self._store.update(data)

    def extend(self, data: Dict[bytes, Any]) -> None:
        """Update the index with the key/value pairs from data, appending
        existing keys with the new data.

        Args:
            data (dict): Geohash codes associated with the values to be
                updated in the database.
        """
        with self._synchronizer:
            self._store.extend(data)

    def box(self, box: geodetic.Box) -> List[Any]:
        """Selection of all data within the defined geographical area

        Args:
            box (pyinterp.geodetic.Box): Bounding box used for data selection.

        Return:
            list: List of data contained in the database for all positions
            located in the selected geographic region.
        """
        result = []
        values = self._store.values(
            list(string.bounding_boxes(box, precision=self._precision)))
        tuple(map(result.extend, values))
        return result

    def __len__(self):
        return len(self._store) - 1

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} precision={self._precision}>"


def init_geohash(store: storage.MutableMapping,
                 precision: int = 3,
                 synchronizer: Optional[lock.Synchronizer] = None) -> GeoHash:
    """Creation of a GeoHash index

    Args:
        store (MutableMapping): Object managing the storage of the index
        precision (int): Accuracy of the index. By default the precision is 3
            characters.
        synchronizer (lock.Synchronizer, optional): Write synchronizer

    Return:
        GeoHash: index handler
    """
    result = GeoHash(store, precision, synchronizer)
    result.set_properties()
    return result


def open_geohash(store: storage.MutableMapping,
                 synchronizer: Optional[lock.Synchronizer] = None) -> GeoHash:
    """Open of a GeoHash index

    Args:
        store (MutableMapping): Object managing the storage of the index
        synchronizer (lock.Synchronizer, optional): Write synchronizer

    Return:
        GeoHash: index handler
    """
    result = GeoHash(store,
                     synchronizer=synchronizer,
                     **GeoHash.get_properties(store))
    return result
