# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import argparse
import concurrent.futures
import multiprocessing
import os
import subprocess
import sys
import sysconfig


def directory_type(value):
    """The option must define a path to a directory"""
    path = os.path.abspath(value)
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError('%r is not a directory' % value)
    return path


def usage():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="Parallel clang-tidy runner")
    parser.add_argument('--include',
                        nargs="+",
                        type=directory_type,
                        help='Add directory to include search path')
    parser.add_argument('--jobs',
                        type=int,
                        default=multiprocessing.cpu_count(),
                        help='number of tidy instances to be run in parallel.')
    parser.add_argument('--fix',
                        action="store_true",
                        help="Apply suggested fixes. Without -fix-errors "
                        "clang-tidy will bail out if any compilation "
                        "errors were found")
    parser.add_argument("--log",
                        type=argparse.FileType("w"),
                        help="path to the file containing the execution log.")
    parser.add_argument(
        "--clang-tidy",
        help='path to the "clang-tidy" program to be executed.',
        default="clang-tidy")
    return parser.parse_args()


def run(program, fix, path, options=""):
    """Launch clang-tidy"""
    args = [
        program, '-checks=*,-llvm-header-guard,-fuchsia-*,-android-*,'
        '-*-magic-numbers,-google-runtime-references,'
        '-cppcoreguidelines-init-variables,'
        '-cppcoreguidelines-pro-bounds-pointer-arithmetic,'
        '-cppcoreguidelines-pro-bounds-array-to-pointer-decay,'
        '-cppcoreguidelines-pro-type-cstyle-cast,'
        '-cppcoreguidelines-pro-type-vararg,'
        '-cppcoreguidelines-pro-bounds-constant-array-index,'
        '-cppcoreguidelines-owning-memory,'
        '-hicpp-*,'
        '-*-non-private-member-variables-in-classes', '-format-style=Google',
        path, '--', options
    ]
    if fix:
        args.insert(2, "-fix")
    process = subprocess.Popen(" ".join(args),
                               shell=True,
                               bufsize=4096,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    stdout, _ = process.communicate()
    stdout = stdout.decode('utf8')
    if process.returncode != 0:
        raise RuntimeError(stdout)
    return " ".join(args) + "\n" + stdout


def main():
    """Main function"""
    args = usage()
    target = []

    # Root of project
    root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.basename(__file__))))

    # Directories to include in search path
    includes = [] if args.include is None else args.include
    includes.insert(0, sysconfig.get_config_var('INCLUDEPY'))
    includes.insert(0, f"{sys.prefix}/include/eigen3")
    includes.insert(0, f"{sys.prefix}/include")
    includes.insert(0, f"{root}/third_party/pybind11/include")
    includes.insert(0, f"{root}/src/pyinterp/core/include")

    # Enumerates files to be processed
    for dirname in [f"{root}/src/pyinterp/core"]:
        for root, dirs, files in os.walk(dirname):
            if 'tests' in dirs:
                dirs.remove('tests')
            for item in files:
                if item.endswith(".cpp") or item.endswith(".hpp"):
                    target.append(os.path.join(root, item))

    # Compiler options
    options = "-std=c++17 " + " ".join((f"-I{item}" for item in includes))

    # Stream used to write in the logbook
    stream = sys.stderr if args.log is None else args.log

    # Finally, we run all code checks
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs) as executor:
        future_to_lint = {
            executor.submit(run, args.clang_tidy, args.fix, path, options):
            path
            for path in target
        }
        for future in concurrent.futures.as_completed(future_to_lint):
            path = future_to_lint[future]
            print(path)
            try:
                stream.write(future.result() + "\n")
                stream.flush()
            except Exception as exc:
                raise RuntimeError('%r generated an exception: %s' %
                                   (path, exc))


if __name__ == "__main__":
    main()
