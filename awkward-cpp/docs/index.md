[![](https://raw.githubusercontent.com/CoffeaTeam/awkward-lite/main/docs-img/logo/logo-300px.png)](https://github.com/CoffeaTeam/awkward-lite)

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they’re not.

This sub-site documents the C++ classes in the `awkward-cpp` library, which `awkward` uses to accelerate some key algorithms.

The main parts of the C++ codebase are:

   * `awkward-cpp/src/cpu-kernels`: processes that iterate over all elements in an array buffer. Any Awkward Array algorithms that can't be expressed in terms of NumPy array manipulations are implemented as Awkward kernels. Individual kernels are [specified in YAML](https://github.com/CoffeaTeam/awkward-lite/blob/main/kernel-specification.yml) and [documented here](../../reference/generated/kernels.html).
   * `awkward-cpp/src/libawkward`: other algorithms that must be implemented in C++. Currently, these are the [ArrayBuilder](../../reference/generated/ak.ArrayBuilder.html) and [AwkwardForth language](../../reference/awkwardforth.html).
   * `awkward-cpp/src/python`: pybind11 bindings for `libawkward` classes and functions.
   * `header-only/awkward`: header-only files intended for inclusion in other C++ projects that create Awkward Arrays. See the [code in GitHub](https://github.com/CoffeaTeam/awkward-lite/tree/main/header-only/awkward).
