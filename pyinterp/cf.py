# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""CF (Climate and Forecast)."""

from __future__ import annotations


class AxisUnit(set):
    """Base class for managing valid CF convention units for coordinate axes.

    This class extends Python's built-in set to store and validate standardized
    units used in Climate and Forecast (CF) conventions for coordinate axes.
    """

    @property
    def units(self) -> set[str]:
        """Retrieve the set of valid units for this axis type.

        Returns:
            A set of strings representing valid CF convention unit names.

        """
        return self


class AxisLatitudeUnit(AxisUnit):
    """Container for valid latitude axis units in CF conventions.

    Stores standardized CF units for latitude coordinates, including variants
    like 'degrees_north', 'degree_N', etc.
    """

    def __init__(self) -> None:
        """Initialize an AxisLatitudeUnit instance."""
        super().__init__(
            (
                "degrees_north",
                "degree_north",
                "degree_N",
                "degrees_N",
                "degreeN",
                "degreesN",
            )
        )


class AxisLongitudeUnit(AxisUnit):
    """Container for valid longitude axis units in CF conventions.

    Stores standardized CF units for longitude coordinates, including variants
    like 'degrees_east', 'degree_E', etc.
    """

    def __init__(self) -> None:
        """Initialize an AxisLongitudeUnit instance."""
        super().__init__(
            (
                "degrees_east",
                "degree_east",
                "degree_E",
                "degrees_E",
                "degreeE",
                "degreesE",
            )
        )


class AxisTimeUnit(AxisUnit):
    """Container for valid time axis units in CF conventions.

    Stores standardized CF units for temporal coordinates, such as 'days'
    and 'seconds'.
    """

    def __init__(self) -> None:
        """Initialize an AxisTimeUnit instance."""
        super().__init__(("days", "seconds"))
