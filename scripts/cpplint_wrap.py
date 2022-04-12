# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pathlib
import subprocess


def main(output="emacs"):
    """main function."""
    # Root of project
    root = pathlib.Path(__file__).parent.parent.joinpath("src")
    args = [
        "cpplint",
        "--filter=-runtime/references,-build/c++11,-build/include_order",
        "--exclude=src/pyinterp/core/tests",
        "--recursive",
        str(root.resolve()),
    ]
    process = subprocess.Popen(" ".join(args),
                               shell=True,
                               bufsize=4096,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    stdout, _ = process.communicate()
    stdout = stdout.decode('utf8')
    if process.returncode != 0:
        raise RuntimeError(stdout)
    print(stdout)


if __name__ == '__main__':
    main()
