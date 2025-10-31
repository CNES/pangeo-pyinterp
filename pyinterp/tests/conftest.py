# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Pytest configuration file."""
import pytest

def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line options to pytest."""
    parser.addoption('--visualize', action='store_true', default=False)
    parser.addoption('--dump', action='store_true', default=False)
    parser.addoption('--measure-coverage', action='store_true', default=False)
