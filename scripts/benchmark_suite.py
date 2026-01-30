#!/usr/bin/env python3
"""Performance Benchmark Suite for pyinterp Interpolators.
=======================================================

This script benchmarks the performance of various interpolation methods
in pyinterp to measure and compare execution times across different
configurations.

Usage:
    python benchmark_pyinterp.py [--output results.json] [--iterations 5]

Output formats:
    - Console summary with timing statistics
    - JSON file for further analysis
    - Markdown table for documentation

"""  # noqa: D205

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

import numpy as np

# Import pyinterp components
import pyinterp
from pyinterp import core
from pyinterp.core.config import fill as fill_config
from pyinterp.core.config import geometric, windowed
from pyinterp.core.config import rtree as rtree_config


if TYPE_CHECKING:
    from pathlib import Path

# Percentage of missing values in gap-filling benchmarks
GAP_FILLING_THRESHOLD = 0.1

# Global random number generator
RNG = np.random.default_rng()


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    grid_shape: tuple[int, ...]
    num_points: int
    iterations: int
    times: list[float] = field(default_factory=list)

    @property
    def mean_time(self) -> float:
        """Mean time recorded."""
        return statistics.mean(self.times) if self.times else 0.0

    @property
    def std_time(self) -> float:
        """Standard deviation of times recorded."""
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def min_time(self) -> float:
        """Minimum time recorded."""
        return min(self.times) if self.times else 0.0

    @property
    def max_time(self) -> float:
        """Maximum time recorded."""
        return max(self.times) if self.times else 0.0

    @property
    def throughput(self) -> float:
        """Points per second."""
        return self.num_points / self.mean_time if self.mean_time > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "grid_shape": self.grid_shape,
            "num_points": self.num_points,
            "iterations": self.iterations,
            "mean_time_ms": self.mean_time * 1000,
            "std_time_ms": self.std_time * 1000,
            "min_time_ms": self.min_time * 1000,
            "max_time_ms": self.max_time * 1000,
            "throughput_pts_per_sec": self.throughput,
        }


class GridFactory:
    """Factory for creating test grids of various dimensions."""

    @staticmethod
    def create_axis(
        start: float, stop: float, num: int, period: float | None = None
    ) -> core.Axis:
        """Create a numeric axis."""
        values = np.linspace(start, stop, num, dtype=np.float64)
        return core.Axis(values, period=period)

    @staticmethod
    def create_temporal_axis(
        start: str, periods: int, freq: str = "D"
    ) -> core.TemporalAxis:
        """Create a temporal axis."""
        dtype = np.dtype(f"datetime64[{freq}]")
        dates = np.arange(  # type: ignore[call-overload]
            start,  # type: ignore[arg-type]
            periods,
            dtype=dtype,
        )
        return core.TemporalAxis(dates)

    @staticmethod
    def create_grid_2d(
        nx: int = 360, ny: int = 180, periodic_x: bool = True
    ) -> core.Grid2D:
        """Create a 2D grid (typical lon/lat grid)."""
        x = GridFactory.create_axis(
            0, 360, nx, period=360.0 if periodic_x else None
        )
        y = GridFactory.create_axis(-90, 90, ny)
        # Create synthetic data (e.g., smooth function)
        xx, yy = np.meshgrid(np.linspace(0, 360, nx), np.linspace(-90, 90, ny))
        data = np.sin(np.radians(xx)) * np.cos(np.radians(yy))
        return core.Grid(x, y, data.T.astype(np.float64))

    @staticmethod
    def create_grid_3d(
        nx: int = 180, ny: int = 90, nz: int = 50, periodic_x: bool = True
    ) -> core.Grid3D:
        """Create a 3D grid (lon/lat/depth or lon/lat/time)."""
        x = GridFactory.create_axis(
            0, 360, nx, period=360.0 if periodic_x else None
        )
        y = GridFactory.create_axis(-90, 90, ny)
        z = GridFactory.create_axis(0, 1000, nz)  # e.g., depth in meters
        # Create synthetic 3D data
        data = RNG.standard_normal((nx, ny, nz)).astype(np.float64)
        return core.Grid(x, y, z, data)

    @staticmethod
    def create_grid_4d(
        nx: int = 90,
        ny: int = 45,
        nz: int = 20,
        nu: int = 10,
        periodic_x: bool = True,
    ) -> core.Grid4D:
        """Create a 4D grid (lon/lat/depth/time)."""
        x = GridFactory.create_axis(
            0, 360, nx, period=360.0 if periodic_x else None
        )
        y = GridFactory.create_axis(-90, 90, ny)
        z = GridFactory.create_axis(0, 1000, nz)
        u = GridFactory.create_axis(0, 100, nu)
        data = RNG.standard_normal((nx, ny, nz, nu)).astype(np.float64)
        return core.Grid(x, y, z, u, data)


class PointGenerator:
    """Generate random interpolation points."""

    @staticmethod
    def generate_2d(
        num_points: int,
        x_range: tuple[float, float] = (0, 360),
        y_range: tuple[float, float] = (-90, 90),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate random 2D points."""
        x = RNG.uniform(x_range[0], x_range[1], num_points).astype(np.float64)
        y = RNG.uniform(y_range[0], y_range[1], num_points).astype(np.float64)
        return x, y

    @staticmethod
    def generate_3d(
        num_points: int,
        x_range: tuple[float, float] = (0, 360),
        y_range: tuple[float, float] = (-90, 90),
        z_range: tuple[float, float] = (0, 1000),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random 3D points."""
        x, y = PointGenerator.generate_2d(num_points, x_range, y_range)
        z = RNG.uniform(z_range[0], z_range[1], num_points).astype(np.float64)
        return x, y, z

    @staticmethod
    def generate_4d(
        num_points: int,
        x_range: tuple[float, float] = (0, 360),
        y_range: tuple[float, float] = (-90, 90),
        z_range: tuple[float, float] = (0, 1000),
        u_range: tuple[float, float] = (0, 100),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate random 4D points."""
        x, y, z = PointGenerator.generate_3d(
            num_points, x_range, y_range, z_range
        )
        u = RNG.uniform(u_range[0], u_range[1], num_points).astype(np.float64)
        return x, y, z, u


class InterpolatorBenchmark:
    """Benchmark suite for pyinterp interpolators."""

    def __init__(self, iterations: int = 5, num_threads: int = 0) -> None:
        """Initialize the benchmark suite."""
        self.iterations = iterations
        self.num_threads = num_threads
        self.results: list[BenchmarkResult] = []

    def _time_function(
        self, func: Callable, *args: object, **kwargs: object
    ) -> float:
        """Time a single function call."""
        start = time.perf_counter()
        func(*args, **kwargs)
        return time.perf_counter() - start

    def _run_benchmark(
        self,
        name: str,
        func: Callable,
        grid_shape: tuple[int, ...],
        num_points: int,
        *args: object,
        **kwargs: object,
    ) -> BenchmarkResult:
        """Run a benchmark multiple times and collect results."""
        result = BenchmarkResult(
            name=name,
            grid_shape=grid_shape,
            num_points=num_points,
            iterations=self.iterations,
        )

        # Warm-up run
        func(*args, **kwargs)

        # Timed runs
        for _ in range(self.iterations):
            elapsed = self._time_function(func, *args, **kwargs)
            result.times.append(elapsed)

        self.results.append(result)
        return result

    # =========================================================================
    # 2D Interpolation Benchmarks
    # =========================================================================

    def benchmark_bivariate_geometric(
        self,
        grid: core.Grid2D,
        x: np.ndarray,
        y: np.ndarray,
        method_name: str,
        config: geometric.Bivariate,
    ) -> BenchmarkResult:
        """Benchmark geometric bivariate interpolation."""
        config = config.with_num_threads(self.num_threads)

        return self._run_benchmark(
            name=f"bivariate_geometric_{method_name}",
            func=core.bivariate,
            grid_shape=grid.shape,
            num_points=len(x),
            grid=grid,
            x=x,
            y=y,
            config=config,
        )

    def benchmark_bivariate_windowed(
        self,
        grid: core.Grid2D,
        x: np.ndarray,
        y: np.ndarray,
        method_name: str,
        config: windowed.Bivariate,
    ) -> BenchmarkResult:
        """Benchmark windowed bivariate interpolation."""
        config = config.with_num_threads(self.num_threads)

        return self._run_benchmark(
            name=f"bivariate_windowed_{method_name}",
            func=core.bivariate,
            grid_shape=grid.shape,
            num_points=len(x),
            grid=grid,
            x=x,
            y=y,
            config=config,
        )

    def run_2d_benchmarks(
        self, grid_sizes: list[tuple[int, int]], point_counts: list[int]
    ) -> None:
        """Run all 2D interpolation benchmarks."""
        print("\n" + "=" * 70)
        print("2D INTERPOLATION BENCHMARKS")
        print("=" * 70)

        # Geometric methods
        geometric_methods = {
            "bilinear": geometric.Bivariate.bilinear(),
            "nearest": geometric.Bivariate.nearest(),
            "idw": geometric.Bivariate.idw(),
        }

        # Windowed methods (spline-based)
        windowed_methods = {
            "bicubic": windowed.Bivariate.bicubic(),
            "linear": windowed.Bivariate.linear(),
            "c_spline": windowed.Bivariate.c_spline(),
            "akima": windowed.Bivariate.akima(),
            "steffen": windowed.Bivariate.steffen(),
            "polynomial": windowed.Bivariate.polynomial(),
        }

        for nx, ny in grid_sizes:
            print(f"\n--- Grid size: {nx} x {ny} ---")
            grid = GridFactory.create_grid_2d(nx, ny)

            for num_points in point_counts:
                print(f"\n  Points: {num_points:,}")
                x, y = PointGenerator.generate_2d(num_points)

                config: object

                # Benchmark geometric methods
                for name, config in geometric_methods.items():
                    result = self.benchmark_bivariate_geometric(
                        grid, x, y, name, config
                    )
                    print(
                        f"    {name:20s}: {result.mean_time * 1000:8.2f} ms "
                        f"(±{result.std_time * 1000:.2f} ms) | "
                        f"{result.throughput / 1e6:.2f} Mpts/s"
                    )

                # Benchmark windowed methods
                for name, config in windowed_methods.items():
                    result = self.benchmark_bivariate_windowed(
                        grid, x, y, name, config
                    )
                    print(
                        f"    {name:20s}: {result.mean_time * 1000:8.2f} ms "
                        f"(±{result.std_time * 1000:.2f} ms) | "
                        f"{result.throughput / 1e6:.2f} Mpts/s"
                    )

    # =========================================================================
    # 3D Interpolation Benchmarks
    # =========================================================================

    def benchmark_trivariate_geometric(
        self,
        grid: core.Grid3D,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        method_name: str,
        config: geometric.Trivariate,
    ) -> BenchmarkResult:
        """Benchmark geometric trivariate interpolation."""
        config = config.with_num_threads(self.num_threads)

        return self._run_benchmark(
            name=f"trivariate_geometric_{method_name}",
            func=core.trivariate,
            grid_shape=grid.shape,
            num_points=len(x),
            grid=grid,
            x=x,
            y=y,
            z=z,
            config=config,
        )

    def benchmark_trivariate_windowed(
        self,
        grid: core.Grid3D,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        method_name: str,
        config: windowed.Trivariate,
    ) -> BenchmarkResult:
        """Benchmark windowed trivariate interpolation."""
        config = config.with_num_threads(self.num_threads)

        return self._run_benchmark(
            name=f"trivariate_windowed_{method_name}",
            func=core.trivariate,
            grid_shape=grid.shape,
            num_points=len(x),
            grid=grid,
            x=x,
            y=y,
            z=z,
            config=config,
        )

    def run_3d_benchmarks(
        self, grid_sizes: list[tuple[int, int, int]], point_counts: list[int]
    ) -> None:
        """Run all 3D interpolation benchmarks."""
        print("\n" + "=" * 70)
        print("3D INTERPOLATION BENCHMARKS")
        print("=" * 70)

        geometric_methods = {
            "bilinear": geometric.Trivariate.bilinear(),
            "nearest": geometric.Trivariate.nearest(),
            "idw": geometric.Trivariate.idw(),
        }

        windowed_methods = {
            "bicubic": windowed.Trivariate.bicubic(),
            "linear": windowed.Trivariate.linear(),
            "c_spline": windowed.Trivariate.c_spline(),
            "akima": windowed.Trivariate.akima(),
        }

        for nx, ny, nz in grid_sizes:
            print(f"\n--- Grid size: {nx} x {ny} x {nz} ---")
            grid = GridFactory.create_grid_3d(nx, ny, nz)

            for num_points in point_counts:
                print(f"\n  Points: {num_points:,}")
                x, y, z = PointGenerator.generate_3d(num_points)

                config: object

                for name, config in geometric_methods.items():
                    result = self.benchmark_trivariate_geometric(
                        grid, x, y, z, name, config
                    )
                    print(
                        f"    {name:20s}: {result.mean_time * 1000:8.2f} ms "
                        f"(±{result.std_time * 1000:.2f} ms) | "
                        f"{result.throughput / 1e6:.2f} Mpts/s"
                    )

                for name, config in windowed_methods.items():
                    result = self.benchmark_trivariate_windowed(
                        grid, x, y, z, name, config
                    )
                    print(
                        f"    {name:20s}: {result.mean_time * 1000:8.2f} ms "
                        f"(±{result.std_time * 1000:.2f} ms) | "
                        f"{result.throughput / 1e6:.2f} Mpts/s"
                    )

    # =========================================================================
    # 4D Interpolation Benchmarks
    # =========================================================================

    def benchmark_quadrivariate_geometric(
        self,
        grid: core.Grid4D,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        method_name: str,
        config: geometric.Quadrivariate,
    ) -> BenchmarkResult:
        """Benchmark geometric quadrivariate interpolation."""
        config = config.with_num_threads(self.num_threads)

        return self._run_benchmark(
            name=f"quadrivariate_geometric_{method_name}",
            func=core.quadrivariate,
            grid_shape=grid.shape,
            num_points=len(x),
            grid=grid,
            x=x,
            y=y,
            z=z,
            u=u,
            config=config,
        )

    def benchmark_quadrivariate_windowed(
        self,
        grid: core.Grid4D,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        method_name: str,
        config: windowed.Quadrivariate,
    ) -> BenchmarkResult:
        """Benchmark windowed quadrivariate interpolation."""
        config = config.with_num_threads(self.num_threads)

        return self._run_benchmark(
            name=f"quadrivariate_windowed_{method_name}",
            func=core.quadrivariate,
            grid_shape=grid.shape,
            num_points=len(x),
            grid=grid,
            x=x,
            y=y,
            z=z,
            u=u,
            config=config,
        )

    def run_4d_benchmarks(
        self,
        grid_sizes: list[tuple[int, int, int, int]],
        point_counts: list[int],
    ) -> None:
        """Run all 4D interpolation benchmarks."""
        print("\n" + "=" * 70)
        print("4D INTERPOLATION BENCHMARKS")
        print("=" * 70)

        geometric_methods = {
            "bilinear": geometric.Quadrivariate.bilinear(),
            "nearest": geometric.Quadrivariate.nearest(),
            "idw": geometric.Quadrivariate.idw(),
        }

        windowed_methods = {
            "bicubic": windowed.Quadrivariate.bicubic(),
            "linear": windowed.Quadrivariate.linear(),
        }

        for nx, ny, nz, nu in grid_sizes:
            print(f"\n--- Grid size: {nx} x {ny} x {nz} x {nu} ---")
            grid = GridFactory.create_grid_4d(nx, ny, nz, nu)

            for num_points in point_counts:
                print(f"\n  Points: {num_points:,}")
                x, y, z, u = PointGenerator.generate_4d(num_points)

                config: object

                for name, config in geometric_methods.items():
                    result = self.benchmark_quadrivariate_geometric(
                        grid, x, y, z, u, name, config
                    )
                    print(
                        f"    {name:20s}: {result.mean_time * 1000:8.2f} ms "
                        f"(±{result.std_time * 1000:.2f} ms) | "
                        f"{result.throughput / 1e6:.2f} Mpts/s"
                    )

                for name, config in windowed_methods.items():
                    result = self.benchmark_quadrivariate_windowed(
                        grid, x, y, z, u, name, config
                    )
                    print(
                        f"    {name:20s}: {result.mean_time * 1000:8.2f} ms "
                        f"(±{result.std_time * 1000:.2f} ms) | "
                        f"{result.throughput / 1e6:.2f} Mpts/s"
                    )

    # =========================================================================
    # RTree Benchmarks
    # =========================================================================

    def run_rtree_benchmarks(
        self, data_sizes: list[int], query_sizes: list[int]
    ) -> None:
        """Run RTree spatial interpolation benchmarks."""
        print("\n" + "=" * 70)
        print("RTREE SPATIAL INTERPOLATION BENCHMARKS")
        print("=" * 70)

        for data_size in data_sizes:
            print(f"\n--- Data points: {data_size:,} ---")

            # Generate random spatial data
            lons = RNG.uniform(0, 360, data_size).astype(np.float64)
            lats = RNG.uniform(-90, 90, data_size).astype(np.float64)
            values = RNG.standard_normal(data_size).astype(np.float64)
            coords = np.column_stack([lons, lats])

            # Create and populate RTree
            rtree = core.RTree3D()
            rtree.packing(coords, values)

            for query_size in query_sizes:
                print(f"\n  Query points: {query_size:,}")
                query_lons = RNG.uniform(0, 360, query_size).astype(np.float64)
                query_lats = RNG.uniform(-90, 90, query_size).astype(
                    np.float64
                )
                query_coords = np.column_stack([query_lons, query_lats])

                # IDW
                idw_config = rtree_config.InverseDistanceWeighting()
                idw_config = idw_config.with_k(11).with_num_threads(
                    self.num_threads
                )
                result = self._run_benchmark(
                    name="rtree_idw",
                    func=rtree.inverse_distance_weighting,
                    grid_shape=(data_size,),
                    num_points=query_size,
                    coordinates=query_coords,
                    config=idw_config,
                )
                print(
                    "    IDW:                 "
                    f"{result.mean_time * 1000:8.2f} ms "
                    f"(±{result.std_time * 1000:.2f} ms) | "
                    f"{result.throughput / 1e6:.2f} Mpts/s"
                )

                # RBF
                rbf_config = rtree_config.RadialBasisFunction()
                rbf_config = (
                    rbf_config.with_k(11)
                    .with_rbf(rtree_config.RBFKernel.GAUSSIAN)
                    .with_num_threads(self.num_threads)
                )
                result = self._run_benchmark(
                    name="rtree_rbf_gaussian",
                    func=rtree.radial_basis_function,
                    grid_shape=(data_size,),
                    num_points=query_size,
                    coordinates=query_coords,
                    config=rbf_config,
                )
                print(
                    "    RBF (Gaussian):      "
                    f"{result.mean_time * 1000:8.2f} ms "
                    f"(±{result.std_time * 1000:.2f} ms) | "
                    f"{result.throughput / 1e6:.2f} Mpts/s"
                )

                # Window function
                wf_config = rtree_config.InterpolationWindow()
                wf_config = (
                    wf_config.with_k(11)
                    .with_wf(rtree_config.WindowKernel.LANCZOS)
                    .with_num_threads(self.num_threads)
                )
                result = self._run_benchmark(
                    name="rtree_window_lanczos",
                    func=rtree.window_function,
                    grid_shape=(data_size,),
                    num_points=query_size,
                    coordinates=query_coords,
                    config=wf_config,
                )
                print(
                    "    Window (Lanczos):    "
                    f"{result.mean_time * 1000:8.2f} ms "
                    f"(±{result.std_time * 1000:.2f} ms) | "
                    f"{result.throughput / 1e6:.2f} Mpts/s"
                )

    # =========================================================================
    # Fill/Gap-filling Benchmarks
    # =========================================================================

    def run_fill_benchmarks(self, grid_sizes: list[tuple[int, int]]) -> None:
        """Run gap-filling algorithm benchmarks."""
        print("\n" + "=" * 70)
        print("GAP-FILLING BENCHMARKS")
        print("=" * 70)

        for nx, ny in grid_sizes:
            print(f"\n--- Grid size: {nx} x {ny} ---")

            # Create grid with ~10% missing values
            data = RNG.standard_normal((nx, ny)).astype(np.float64)
            mask = RNG.random((nx, ny)) < GAP_FILLING_THRESHOLD
            data[mask] = np.nan

            # Gauss-Seidel
            gs_config = fill_config.GaussSeidel()
            gs_config = (
                gs_config.with_max_iterations(100)
                .with_epsilon(1e-6)
                .with_num_threads(self.num_threads)
            )
            data_gs = data.copy()
            result = self._run_benchmark(
                name="fill_gauss_seidel",
                func=core.fill.gauss_seidel,
                grid_shape=(nx, ny),
                num_points=nx * ny,
                grid=data_gs,
                config=gs_config,
            )
            print(
                f"  Gauss-Seidel:    {result.mean_time * 1000:8.2f} ms "
                f"(±{result.std_time * 1000:.2f} ms)"
            )

    # =========================================================================
    # Output Generation
    # =========================================================================

    def export_json(self, filepath: str | Path) -> None:
        """Export results to JSON file."""
        data = {
            "metadata": {
                "pyinterp_version": getattr(
                    pyinterp, "__version__", "unknown"
                ),
                "iterations": self.iterations,
                "num_threads": self.num_threads,
            },
            "results": [r.to_dict() for r in self.results],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults exported to: {filepath}")

    def generate_markdown_table(self) -> str:
        """Generate a Markdown table from results."""
        lines = [
            "| Method | Grid Shape | Points | Mean (ms) | Std (ms) |"
            " Throughput (Mpts/s) |",
            "|--------|------------|--------|-----------|----------|"
            "---------------------|",
        ]

        for r in self.results:
            shape_str = "x".join(map(str, r.grid_shape))
            lines.append(
                f"| {r.name} | {shape_str} | {r.num_points:,} | "
                f"{r.mean_time * 1000:.2f} | {r.std_time * 1000:.2f} | "
                f"{r.throughput / 1e6:.2f} |"
            )

        return "\n".join(lines)

    def generate_rst_table(self) -> str:
        """Generate a reStructuredText table from results."""
        # Calculate column widths
        headers = [
            "Method",
            "Grid Shape",
            "Points",
            "Mean (ms)",
            "Std (ms)",
            "Throughput (Mpts/s)",
        ]

        rows = []
        for r in self.results:
            shape_str = "x".join(map(str, r.grid_shape))
            rows.append(
                [
                    r.name,
                    shape_str,
                    f"{r.num_points:,}",
                    f"{r.mean_time * 1000:.2f}",
                    f"{r.std_time * 1000:.2f}",
                    f"{r.throughput / 1e6:.2f}",
                ]
            )

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        # Build table
        separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        header_sep = "+" + "+".join("=" * (w + 2) for w in widths) + "+"

        lines = [separator]
        lines.append(
            "| "
            + " | ".join(
                h.ljust(w) for h, w in zip(headers, widths, strict=False)
            )
            + " |"
        )
        lines.append(header_sep)

        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    c.ljust(w) for c, w in zip(row, widths, strict=False)
                )
                + " |"
            )
            lines.append(separator)

        return "\n".join(lines)


def usage() -> argparse.Namespace:
    """Define command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark pyinterp interpolation performance"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output JSON file path"
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=5,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=0,
        help="Number of threads (0 = auto)",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick benchmark with smaller sizes",
    )
    parser.add_argument(
        "--markdown", "-m", action="store_true", help="Output Markdown table"
    )
    parser.add_argument(
        "--rst",
        "-r",
        action="store_true",
        help="Output reStructuredText table",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments and run benchmarks."""
    args = usage()

    print("=" * 70)
    print("PYINTERP PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)
    print(f"pyinterp version: {getattr(pyinterp, '__version__', 'unknown')}")
    print(f"NumPy version: {np.__version__}")
    print(f"Iterations: {args.iterations}")
    print(f"Threads: {args.threads if args.threads > 0 else 'auto'}")

    benchmark = InterpolatorBenchmark(
        iterations=args.iterations, num_threads=args.threads
    )

    if args.quick:
        # Quick benchmark configuration
        grid_2d_sizes = [(180, 90)]
        grid_3d_sizes = [(90, 45, 20)]
        grid_4d_sizes = [(45, 23, 10, 5)]
        point_counts = [10_000, 100_000]
        rtree_data_sizes = [10_000]
        rtree_query_sizes = [10_000]
        fill_sizes = [(180, 90)]
    else:
        # Full benchmark configuration
        grid_2d_sizes = [(180, 90), (360, 180), (720, 360)]
        grid_3d_sizes = [(180, 90, 50), (360, 180, 100)]
        grid_4d_sizes = [(90, 45, 20, 10)]
        point_counts = [10_000, 100_000, 1_000_000]
        rtree_data_sizes = [10_000, 100_000]
        rtree_query_sizes = [10_000, 100_000]
        fill_sizes = [(360, 180), (720, 360)]

    # Run benchmarks
    benchmark.run_2d_benchmarks(grid_2d_sizes, point_counts)
    benchmark.run_3d_benchmarks(grid_3d_sizes, point_counts)
    benchmark.run_4d_benchmarks(grid_4d_sizes, point_counts)
    benchmark.run_rtree_benchmarks(rtree_data_sizes, rtree_query_sizes)
    benchmark.run_fill_benchmarks(fill_sizes)

    # Output results
    if args.output:
        benchmark.export_json(args.output)

    if args.markdown:
        print("\n" + "=" * 70)
        print("MARKDOWN TABLE")
        print("=" * 70)
        print(benchmark.generate_markdown_table())

    if args.rst:
        print("\n" + "=" * 70)
        print("RST TABLE")
        print("=" * 70)
        print(benchmark.generate_rst_table())

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
