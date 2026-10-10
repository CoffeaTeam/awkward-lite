# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

import pytest

import awkward as ak

cudf = pytest.importorskip("cudf", exc_type=ImportError)


def test_series():
    out = ak.from_cudf(cudf.Series([1, 2, 3]))
    assert isinstance(out, ak.Array)
    assert ak.backend(out) == "cpu"
    assert out.to_list() == [1, 2, 3]


def test_null():
    out = ak.from_cudf(cudf.Series([12, None, 21, 12]))
    assert out.to_list() == [12, None, 21, 12]


def test_jagged():
    out = ak.from_cudf(cudf.Series([[1, 2, 3], [], [3, 4]]))
    assert out.to_list() == [[1, 2, 3], [], [3, 4]]


def test_strings():
    out = ak.from_cudf(cudf.Series(["hey", "hi", None, "hum"]))
    assert out.to_list() == ["hey", "hi", None, "hum"]


def test_dataframe():
    out = ak.from_cudf(cudf.DataFrame({"x": [1, 2, 3], "y": [1.1, 2.2, 3.3]}))
    assert out.fields == ["x", "y"]
    assert out.to_list() == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]


def test_roundtrip():
    array = ak.Array([[{"x": 1, "y": [1.1]}], [], [{"x": 2, "y": []}]])
    assert ak.from_cudf(ak.to_cudf(array)).to_list() == array.to_list()


def test_low_level():
    out = ak.from_cudf(cudf.Series([1, 2, 3]), highlevel=False)
    assert isinstance(out, ak.contents.Content)


def test_other_types_are_rejected():
    with pytest.raises(TypeError, match=r"cudf\.Series or cudf\.DataFrame"):
        ak.from_cudf([1, 2, 3])
