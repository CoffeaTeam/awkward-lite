# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

cudf = pytest.importorskip("cudf", exc_type=ImportError)


def test_jagged():
    arr = ak.Array([[[1, 2, 3], [], [3, 4]], []])
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert out.to_arrow().tolist() == [[[1, 2, 3], [], [3, 4]], []]


def test_nested():
    arr = ak.Array(
        [{"a": 0, "b": 1.0, "c": {"d": 0}}, {"a": 1, "b": 0.0, "c": {"d": 1}}]
    )
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert out.to_arrow().tolist() == [
        {"a": 0, "b": 1.0, "c": {"d": 0}},
        {"a": 1, "b": 0.0, "c": {"d": 1}},
    ]


def test_null():
    arr = ak.Array([12, None, 21, 12])
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert out.to_arrow().tolist() == [12, None, 21, 12]

    # True is valid, LSB order
    arr2 = ak.Array(arr.layout.to_BitMaskedArray(True, True))
    out = ak.to_cudf(arr2)
    assert out.to_arrow().tolist() == [12, None, 21, 12]

    # reversed LSB
    arr3 = ak.Array(arr.layout.to_BitMaskedArray(True, False))
    out = ak.to_cudf(arr3)
    assert out.to_arrow().tolist() == [12, None, 21, 12]

    arr4 = ak.Array([[1, None], None, [3]])
    out = ak.to_cudf(arr4)
    assert out.to_arrow().tolist() == [[1, None], None, [3]]


def test_strings():
    arr = ak.Array(["hey", "hi", "hum"])
    out = ak.to_cudf(arr)
    assert out.to_arrow().tolist() == ["hey", "hi", "hum"]

    arr = ak.Array(["hey", "hi", None, "hum"])
    out = ak.to_cudf(arr)
    assert out.to_arrow().tolist() == ["hey", "hi", None, "hum"]

    arr = ak.Array([["hey", "hi"], [], ["hum"]])
    out = ak.to_cudf(arr)
    assert out.to_arrow().tolist() == [["hey", "hi"], [], ["hum"]]


def test_regular():
    arr = ak.to_regular(ak.Array([[1, 2], [3, 4]]), axis=1)
    out = ak.to_cudf(arr)
    assert out.to_arrow().tolist() == [[1, 2], [3, 4]]

    arr = ak.from_numpy(np.arange(6).reshape(3, 2))
    out = ak.to_cudf(arr)
    assert out.to_arrow().tolist() == [[0, 1], [2, 3], [4, 5]]


def test_categorical():
    arr = ak.str.to_categorical(ak.Array(["a", "b", "a"]))
    out = ak.to_cudf(arr)
    assert isinstance(out.dtype, cudf.CategoricalDtype)
    assert out.to_arrow().tolist() == ["a", "b", "a"]


def test_empty():
    out = ak.to_cudf(ak.Array([[], []]))
    assert out.to_arrow().tolist() == [[], []]
    assert out.dtype.element_type == np.dtype(np.float64)
