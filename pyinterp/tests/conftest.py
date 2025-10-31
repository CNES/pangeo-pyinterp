# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Pytest configuration file."""
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line options to pytest.

    This file serves as the package's root conftest.py. Ensure options are only
    added if they are not already defined by a higher-level conftest.py.
    """

    def add_option_if_not_exists(option_name: str,
                                 **kwargs: str | bool) -> None:
        """Add option only if it doesn't already exist."""
        try:
            parser.addoption(option_name, **kwargs)
        except ValueError:
            pass  # Option already registered

    add_option_if_not_exists('--visualize', action='store_true', default=False)
    add_option_if_not_exists('--dump', action='store_true', default=False)
    add_option_if_not_exists('--measure-coverage',
                             action='store_true',
                             default=False)
