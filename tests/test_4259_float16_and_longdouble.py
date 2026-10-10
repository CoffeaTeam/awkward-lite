# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

from __future__ import annotations

import io
import json

import numpy as np
import pytest

import awkward as ak

# long double has 80 or 128 bits on Linux and is the same as double on
# macOS arm64 and Windows
LONGDOUBLE_IS_WIDER = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant

REDUCERS = [
    (ak.sum, np.sum),
    (ak.prod, np.prod),
    (ak.min, np.min),
    (ak.max, np.max),
    (ak.argmin, np.argmin),
    (ak.argmax, np.argmax),
    (ak.any, np.any),
    (ak.all, np.all),
    (ak.count_nonzero, np.count_nonzero),
]


def values(dtype):
    rng = np.random.default_rng(4259)
    out = rng.integers(1, 5, size=(50, 6)) + rng.normal(size=(50, 6))
    if np.dtype(dtype).kind == "c":
        out = out + 1j * rng.integers(0, 3, size=(50, 6))
    return out.astype(dtype)


def result_dtype(expected):
    # NumPy counts and indexes with intp, which is int32 on 32-bit platforms
    return np.dtype(np.int64) if expected.dtype.kind == "i" else expected.dtype


@pytest.mark.parametrize("dtype", [np.float16, np.longdouble, np.clongdouble])
@pytest.mark.parametrize("ak_reducer, np_reducer", REDUCERS)
def test_reducers(dtype, ak_reducer, np_reducer):
    data = values(dtype)
    array = ak.from_regular(ak.Array(data))

    result = ak.to_numpy(ak_reducer(array, axis=1))
    expected = np_reducer(data, axis=1)
    assert result.dtype == result_dtype(expected)
    if ak_reducer in (ak.sum, ak.prod):
        # the kernels and NumPy do not add and multiply in the same order
        assert np.allclose(
            result, expected, rtol=1e-2 if dtype is np.float16 else 1e-12
        )
    else:
        assert np.array_equal(result, expected)


@pytest.mark.parametrize("dtype", [np.float16, np.longdouble])
def test_sort_and_argsort(dtype):
    data = values(dtype)
    array = ak.from_regular(ak.Array(data))

    result = ak.to_numpy(ak.sort(array, axis=1))
    assert result.dtype == np.dtype(dtype)
    assert np.array_equal(result, np.sort(data, axis=1))

    result = ak.to_numpy(ak.sort(array, axis=1, ascending=False))
    assert np.array_equal(result, np.sort(data, axis=1)[:, ::-1])

    result = ak.to_numpy(ak.argsort(array, axis=1, stable=True))
    assert np.array_equal(result, np.argsort(data, axis=1, kind="stable"))


def test_float16_with_missing_values():
    array = ak.Array(np.array([[3, 1, 2], [5, 4, 6]], dtype=np.float16))
    array = ak.mask(ak.from_regular(array), [[True, False, True], [True, True, True]])

    assert ak.sum(array, axis=1).to_list() == [5, 15]
    assert ak.sort(array, axis=1).to_list() == [[2, 3, None], [4, 5, 6]]
    assert ak.argmax(array, axis=1).to_list() == [0, 2]
    assert str(ak.sort(array, axis=1).type) == "2 * var * ?float16"


def test_float16_axis_none():
    data = values(np.float16)[:2]
    array = ak.Array(data)

    for ak_reducer, np_reducer in REDUCERS:
        result = ak_reducer(array, axis=None)
        expected = np_reducer(data)
        assert np.asarray(result).dtype == result_dtype(np.asarray(expected))
        assert result == pytest.approx(expected, rel=1e-2)


def test_float16_typetracer():
    array = ak.from_regular(ak.Array(values(np.float16)))
    tracer = ak.to_backend(array, "typetracer")

    for function in (ak.sum, ak.min, ak.argmax, ak.any, ak.sort, ak.argsort):
        assert function(tracer, axis=1).type == function(array, axis=1).type


def test_float16_totals_are_rounded_once_as_in_numpy():
    # 60000 + 60000 does not fit in float16, the total does
    data = np.array([[60000, 60000, -60000]], dtype=np.float16)
    array = ak.from_regular(ak.Array(data))
    assert ak.to_numpy(ak.sum(array, axis=1))[0] == np.sum(data) == 60000

    # the same for 300 * 300
    data = np.array([[300, 300, 0.001]], dtype=np.float16)
    array = ak.from_regular(ak.Array(data))
    assert ak.to_numpy(ak.prod(array, axis=1))[0] == np.prod(data)
    assert np.isfinite(np.prod(data))


@pytest.mark.skipif(
    not LONGDOUBLE_IS_WIDER,
    reason="long double is not wider than double on this platform",
)
def test_longdouble_keeps_its_precision():
    # these values are all equal to 1 when they are rounded to float64
    one = np.longdouble(1)
    eps = np.finfo(np.longdouble).eps
    data = np.array([[one + 2 * eps, one, one + eps, one + 3 * eps]])
    assert len(set(data.astype(np.float64).ravel().tolist())) == 1
    array = ak.from_regular(ak.Array(data))

    assert ak.to_numpy(ak.min(array, axis=1))[0] == one
    assert ak.to_numpy(ak.max(array, axis=1))[0] == one + 3 * eps
    assert ak.to_numpy(ak.argmin(array, axis=1))[0] == 1
    assert ak.to_numpy(ak.argmax(array, axis=1))[0] == 3
    assert np.array_equal(ak.to_numpy(ak.sort(array, axis=1)), np.sort(data, axis=1))
    assert ak.to_numpy(ak.argsort(array, axis=1)).tolist() == [[1, 2, 0, 3]]
    assert ak.to_numpy(ak.sum(array, axis=1))[0] == np.sum(data)

    as_complex = ak.from_regular(ak.Array(data.astype(np.clongdouble)))
    assert ak.to_numpy(ak.max(as_complex, axis=1))[0] == one + 3 * eps
    assert ak.to_numpy(ak.sum(as_complex, axis=1))[0] == np.sum(data)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.clongdouble])
def test_complex_numbers_cannot_be_sorted(dtype):
    array = ak.from_regular(ak.Array(values(dtype)))

    with pytest.raises(ValueError, match=r"cannot sort \(ak\.sort\) values of"):
        ak.sort(array, axis=1)
    with pytest.raises(ValueError, match=r"cannot sort \(ak\.argsort\) values of"):
        ak.argsort(array, axis=1)
    with pytest.raises(ValueError, match="cannot find the unique values of"):
        ak._do.unique(array.layout, axis=-1)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.clongdouble])
def test_unique_complex_numbers_without_an_axis(dtype):
    data = np.array([2 + 1j, 1 + 2j, 2 + 1j, 1 - 1j, 1 + 2j], dtype=dtype)

    for array in (ak.Array(data), ak.unflatten(ak.Array(data), [2, 0, 3])):
        result = ak.to_numpy(ak._do.unique(array.layout, axis=None))
        assert result.dtype == np.dtype(dtype)
        assert np.array_equal(result, np.unique(data))
        assert not ak._do.is_unique(array.layout)

    assert ak._do.is_unique(ak.Array(data[2:]).layout)


def test_unique_date_times():
    data = np.array(["2020-01-02", "2020-01-01", "2020-01-02"], "datetime64[s]")
    array = ak.Array(data)

    result = ak.to_numpy(ak._do.unique(array.layout, axis=None))
    assert np.array_equal(result, np.unique(data))

    with pytest.raises(ValueError, match="cannot find the unique values of"):
        ak._do.unique(ak.unflatten(array, [2, 1]).layout, axis=-1)


@pytest.mark.parametrize("dtype", [np.float16, np.longdouble, np.clongdouble])
def test_type_strings_can_be_parsed(dtype):
    array = ak.from_regular(ak.Array(values(dtype)))

    assert ak.types.from_datashape(str(array.type), highlevel=True) == array.type

    name = np.dtype(dtype).name
    as_option = ak.enforce_type(array, f"var * ?{name}")
    assert str(as_option.type) == f"50 * var * ?{name}"
    assert ak.to_numpy(ak.flatten(as_option)).dtype == np.dtype(dtype)


def test_enforce_type_to_float16():
    array = ak.Array(np.array([1.5, 2.5], dtype=np.float32))

    result = ak.enforce_type(array, "float16")
    assert str(result.type) == "2 * float16"
    assert result.to_list() == [1.5, 2.5]


def test_covar_and_corr_keep_long_double():
    x = ak.Array(np.array([[1, 2, 4, 8]], dtype=np.longdouble))
    y = ak.Array(np.array([[1, 3, 2, 9]], dtype=np.longdouble))
    expected = np.cov(np.asarray(x[0]), np.asarray(y[0]), bias=True)

    result = ak.to_numpy(ak.covar(x, y, axis=1))
    assert result.dtype == expected.dtype == np.dtype(np.longdouble)
    assert result[0] == pytest.approx(expected[0, 1])
    assert ak.to_numpy(ak.corr(x, y, axis=1)).dtype == np.dtype(np.longdouble)

    # NumPy makes float64 out of the narrower types
    x, y = ak.values_astype(x, np.float16), ak.values_astype(y, np.float16)
    assert ak.to_numpy(ak.covar(x, y, axis=1)).dtype == np.dtype(np.float64)


def test_from_iter_with_numpy_scalars():
    array = ak.from_iter([np.float16(1.5), np.float32(2.5)])
    assert str(array.type) == "2 * float64"
    assert array.to_list() == [1.5, 2.5]

    array = ak.from_iter([np.complex64(1 + 2j), np.float16(1.5)])
    assert str(array.type) == "2 * complex128"
    assert array.to_list() == [1 + 2j, 1.5 + 0j]


@pytest.mark.parametrize("scalar", [np.longdouble(1.5), np.clongdouble(1.5 + 2j)])
def test_builder_with_long_double_scalars(scalar):
    # the builder holds float64 and complex128 numbers, whatever the type
    # of the NumPy scalar
    builder = ak.ArrayBuilder()
    builder.append(scalar)

    for array in (ak.from_iter([scalar]), builder.snapshot()):
        assert array.to_list() == [scalar]
        assert array.layout.dtype.type in (np.float64, np.complex128)


@pytest.mark.parametrize(
    "dtype, key",
    [
        (np.float16, "float"),
        (np.float16, "float_kind"),
        (np.longdouble, "longfloat"),
        (np.longdouble, "float_kind"),
        (np.clongdouble, "longcomplexfloat"),
        (np.clongdouble, "complex_kind"),
    ],
)
def test_show_with_a_formatter(dtype, key):
    array = ak.Array(np.array([1.5, 2.5], dtype=dtype))

    stream = io.StringIO()
    array.show(formatter={key: lambda x: "<value>"}, stream=stream)
    assert stream.getvalue() == "[<value>,\n <value>]\n"


@pytest.mark.parametrize("dtype", [np.float16, np.longdouble])
def test_show_with_a_precision(dtype):
    array = ak.Array(np.array([1 / 3], dtype=dtype))

    stream = io.StringIO()
    array.show(precision=2, stream=stream)
    assert stream.getvalue() == "[0.33]\n"


def test_json():
    half = ak.from_regular(ak.Array(values(np.float16)))
    assert json.loads(ak.to_json(half)) == half.to_list()

    array = ak.from_regular(ak.Array(values(np.longdouble)))
    assert json.loads(ak.to_json(array)) == array.to_list()

    # JSON is written from Python floats, so it has the nearest double
    third = np.longdouble(1) / np.longdouble(3)
    assert json.loads(ak.to_json(ak.Array(np.array([third])))) == [float(third)]

    array = ak.Array(np.array([third + 2j * third], dtype=np.clongdouble))
    written = ak.to_json(array, complex_record_fields=("real", "imag"))
    assert json.loads(written) == [{"real": float(third), "imag": 2 * float(third)}]


def test_arrow():
    pytest.importorskip("pyarrow")

    half = ak.from_regular(ak.Array(values(np.float16)))
    back = ak.from_arrow(ak.to_arrow(half))
    assert back.type == half.type
    assert back.to_list() == half.to_list()

    array = ak.from_regular(ak.Array(values(np.longdouble)))
    if LONGDOUBLE_IS_WIDER:
        with pytest.raises(TypeError, match="Arrow has no type for float128"):
            ak.to_arrow(array)
    else:
        back = ak.from_arrow(ak.to_arrow(array))
        assert str(back.type) == "50 * var * float64"
        assert back.to_list() == array.to_list()

    array = ak.from_regular(ak.Array(values(np.clongdouble)))
    with pytest.raises(TypeError, match="Arrow has no type for complex"):
        ak.to_arrow(array)


def test_numba():
    numba = pytest.importorskip("numba")
    pytest.importorskip("awkward.numba").register_and_check()

    @numba.njit
    def first(array):
        return array[0][0]

    @numba.njit
    def without_first(array):
        return array[1:]

    # float16 arrays can go through a compiled function, but not be read
    half = ak.from_regular(ak.Array(values(np.float16)))
    result = without_first(half)
    assert result.type == half[1:].type
    assert result.to_list() == half[1:].to_list()
    with pytest.raises(TypeError, match="Numba cannot read float16 values"):
        first(half)

    array = ak.from_regular(ak.Array(values(np.longdouble)))
    if LONGDOUBLE_IS_WIDER:
        with pytest.raises(TypeError, match="Numba has no long double type"):
            without_first(array)
    else:
        assert first(array) == array[0][0]


def test_cpp_value_types():
    cling = pytest.importorskip("awkward._connect.cling")

    def value_type(dtype):
        form = ak.Array(np.zeros(3, dtype=dtype)).layout.form
        return cling.togenerator(form, flatlist_as_rvec=False).value_type()

    if LONGDOUBLE_IS_WIDER:
        assert value_type(np.longdouble) == "long double"
        assert value_type(np.clongdouble) == "std::complex<long double>"
    else:
        assert value_type(np.longdouble) == "double"
        assert value_type(np.clongdouble) == "std::complex<double>"

    with pytest.raises(TypeError, match=r"C\+\+ has no half precision type"):
        value_type(np.float16)


@pytest.mark.parametrize("dtype", [np.longdouble, np.clongdouble])
def test_rdataframe(dtype):
    pytest.importorskip("ROOT")

    array = ak.from_regular(ak.Array(values(dtype)))
    data_frame = ak.to_rdataframe({"x": array})
    if LONGDOUBLE_IS_WIDER:
        assert "long double" in str(data_frame.GetColumnType("x"))

    back = ak.from_rdataframe(data_frame, columns=("x",))["x"]
    assert back.type == array.type
    assert ak.all(back == array, axis=None)


def test_rdataframe_long_double_column():
    ROOT = pytest.importorskip("ROOT")

    data_frame = ROOT.RDataFrame(3).Define("x", "1.0L / 3.0L + rdfentry_")
    assert str(data_frame.GetColumnType("x")) == "long double"

    result = ak.to_numpy(ak.from_rdataframe(data_frame, columns=("x",))["x"])
    assert result.dtype == np.dtype(np.longdouble)
    assert result[0] == np.longdouble(1) / np.longdouble(3)


def test_rdataframe_float16():
    pytest.importorskip("ROOT")

    array = ak.from_regular(ak.Array(values(np.float16)))
    with pytest.raises(TypeError, match=r"C\+\+ has no half precision type"):
        ak.to_rdataframe({"x": array})


def test_long_double_form_from_an_awkward1_pickle():
    form = ak.forms.NumpyForm.__new__(ak.forms.NumpyForm)
    state = (False, None, None, (), np.dtype(np.longdouble).itemsize, "g")

    if hasattr(np, "float128"):
        form.__setstate__(state)
        assert form.primitive == "float128"
    else:
        with pytest.raises(TypeError, match="unrecognized primitive: 'float128'"):
            form.__setstate__(state)
