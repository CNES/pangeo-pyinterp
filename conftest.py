# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import os
import pathlib
import sys
import sysconfig

# Check Python requirement
MAJOR = sys.version_info[0]
MINOR = sys.version_info[1]

# Working directory
WORKING_DIRECTORY = pathlib.Path(__file__).parent.absolute()


def build_dirname(extname=None):
    """Returns the name of the build directory"""
    extname = '' if extname is None else os.sep.join(extname.split(".")[:-1])
    return pathlib.Path(
        WORKING_DIRECTORY, "build",
        "lib.%s-%d.%d" % (sysconfig.get_platform(), MAJOR, MINOR), extname)


sys.path.insert(0, str(build_dirname().resolve()))
