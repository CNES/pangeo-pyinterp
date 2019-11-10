import unittest
import numpy as np
import pyinterp
import pyinterp.interface
import pyinterp.grid


class Interface(unittest.TestCase):
    def test_core_class_suffix(self):
        lon = pyinterp.Axis(0, 360, 1, is_circle=True)
        lat = pyinterp.Axis(-80, 80, 1, is_circle=False)
        for dtype in [
                "float64", "float32", "int64", "uint64", "int32", "uint32",
                "int16", "uint16", "int8", "uint8"
        ]:
            matrix, _ = np.meshgrid(lon[:], lat[:])
            self.assertIsInstance(
                pyinterp.Grid2D(lon, lat,
                                matrix.astype(dtype=getattr(np, dtype))),
                pyinterp.Grid2D)

        with self.assertRaises(ValueError):
            pyinterp.Grid2D(lon, lat, matrix.astype(np.complex))

    def test__core_function_suffix(self):
        with self.assertRaises(TypeError):
            pyinterp.interface._core_function_suffix(1)

        lon = pyinterp.Axis(0, 360, 1, is_circle=True)
        lat = pyinterp.Axis(-80, 80, 1, is_circle=False)
        matrix, _ = np.meshgrid(lon[:], lat[:])
        self.assertEqual(
            pyinterp.interface._core_function_suffix(
                pyinterp.core.Grid2DFloat64(lon, lat, matrix)), "float64")
        self.assertEqual(
            pyinterp.interface._core_function_suffix(
                pyinterp.core.Grid2DFloat32(lon, lat, matrix)), "float32")


class Grid2D(unittest.TestCase):
    def test_core_variate_interpolator(self):
        lon = pyinterp.Axis(0, 360, 1, is_circle=True)
        lat = pyinterp.Axis(-80, 80, 1, is_circle=False)
        matrix, _ = np.meshgrid(lon[:], lat[:])

        grid = pyinterp.Grid2D(lon, lat, matrix)

        with self.assertRaises(TypeError):
            pyinterp.grid._core_variate_interpolator(None, "_")

        with self.assertRaises(ValueError):
            pyinterp.grid._core_variate_interpolator(grid, '_')


if __name__ == "__main__":
    unittest.main()
