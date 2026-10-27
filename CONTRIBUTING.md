Thank you for your interest in this project! Please refer to the following
sections on how to contribute code and bug reports.

## Reporting Bugs

At the moment, this project is run in the spare time of a single person
(Frederic Briol) with very limited resources for issue tracker tickets. Thus,
before submitting a question or bug report, please take a moment of your time
and ensure that your issue isn't already discussed in the project documentation
provided at https://cnes.github.io/pangeo-pyinterp.

Assuming that you have identified a previously unknown problem or an important
question, it's essential that you submit a self-contained and minimal piece of
code that reproduces the problem. In other words: no external dependencies,
isolate the function(s) that cause breakage, submit matched and complete Python
snippets that can be easily run on my end.

## Pull Requests

Contributions are submitted, reviewed, and accepted using Github pull requests.
Please refer to [this
article](https://help.github.com/articles/using-pull-requests) for details and
adhere to the following rules to make the process as smooth as possible:
* Make a new branch for every feature you're working on.
* Make small and clean pull requests that are easy to review but make sure they
  do add value by themselves.
* Add tests for any new functionality and run the [test
  suite](https://cnes.github.io/pangeo-pyinterp/setup/build.html#testing)
  to ensure that no existing features break.
* Install the [pre-commit](https://pre-commit.com/) hooks with
  `pre-commit install`: they format and check the Python, C++ and CMake code
  on each commit. The C++ code is also checked by clang-tidy, with the checks
  defined in `.clang-tidy`.

The [build
guide](https://cnes.github.io/pangeo-pyinterp/setup/build.html) explains how to
set up a development environment, speed up the build, and build the
documentation.

## Licensing of contributions

`pyinterp` is provided under a BSD-style license that can be found in
the `LICENSE` file. By using, distributing, or contributing to this project, you
agree to the terms and conditions of this license. You are under no obligation
whatsoever to provide any bug fixes, patches, or upgrades to the features,
functionality or performance of the source code ("Enhancements") to anyone;
however, if you choose to make your Enhancements available either publicly, or
directly to the author of this software, without imposing a separate written
license agreement for such Enhancements, then you hereby grant the following
license: a non-exclusive, royalty-free perpetual license to install, use,
modify, prepare derivative works, incorporate into other computer software,
distribute, and sublicense such enhancements or derivative works thereof, in
binary and source code form.
