# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

from __future__ import annotations

import ctypes

import awkward_cpp
import numpy as np
import pytest

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._backends.typetracer import TypeTracerBackend
from awkward._kernels import CTypesKernel, NumpyKernel

KEY = ("awkward_ByteMaskedArray_numnull", np.int64, np.int8)
MISSING_KEY = ("awkward_not_a_kernel", np.int64)


def fresh_backend(cls):
    # the backends are singletons, so reach past `instance()` for a cold cache
    backend = object.__new__(cls)
    backend.__init__()
    return backend


@pytest.mark.parametrize("cls", [NumpyBackend, TypeTracerBackend])
def test_fresh_backend_has_empty_kernel_cache(cls):
    backend = fresh_backend(cls)
    assert backend._kernels == {}
    assert backend[KEY] is backend[KEY]
    assert list(backend._kernels) == [KEY]


def test_failed_lookup_is_not_cached():
    backend = fresh_backend(NumpyBackend)
    for _ in range(2):
        with pytest.raises(KeyError):
            backend[MISSING_KEY]
    assert MISSING_KEY not in backend._kernels


def test_ctypes_kernel_pointer_of_is_abstract():
    kernel = CTypesKernel(awkward_cpp.cpu_kernels.kernel[KEY], KEY)
    with pytest.raises(NotImplementedError):
        kernel._pointer_of(np.arange(3))


def test_numpy_pointer_of():
    kernel = NumpyKernel(awkward_cpp.cpu_kernels.kernel[KEY], KEY)

    array = np.arange(3, dtype=np.int64)
    assert kernel._pointer_of(array) == array.ctypes.data

    # a 0-d array is passed through untouched
    scalar = np.array(5, dtype=np.int64)
    assert kernel._pointer_of(scalar) is scalar

    assert isinstance(kernel._pointer_of((ctypes.c_int64 * 3)()), ctypes.c_void_p)

    with pytest.raises(AssertionError, match="Only NumPy buffers"):
        kernel._pointer_of([1, 2, 3])


def test_kernels_still_give_the_same_answers():
    array = ak.Array([[1, 2, 3], [], [4, 5]])

    assert ak.num(array).to_list() == [3, 0, 2]
    assert ak.flatten(array).to_list() == [1, 2, 3, 4, 5]
    assert array[:, 1:].to_list() == [[2, 3], [], [5]]
    assert ak.to_list(array[[0, 2]]) == [[1, 2, 3], [4, 5]]

    typetracer = array.layout.to_typetracer(forget_length=True)
    assert str(ak.num(ak.Array(typetracer)).type) == "## * int64"
