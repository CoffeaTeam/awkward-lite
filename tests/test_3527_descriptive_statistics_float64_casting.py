# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_int32_overflow():
    np.random.seed(42)
    x = np.random.randint(2**21, 2**22, size=1000, dtype=np.int32)
    y = np.random.randint(2**21, 2**22, size=1000, dtype=np.int32)

    np.testing.assert_allclose(np.sum(x), ak.sum(x))
    np.testing.assert_allclose(np.mean(x), ak.mean(x))
    np.testing.assert_allclose(np.var(x), ak.var(x))
    np.testing.assert_allclose(np.std(x), ak.std(x))
    np.testing.assert_allclose(np.cov(x, y, ddof=0)[0][1], ak.covar(x, y))
    np.testing.assert_allclose(np.corrcoef(x, y)[0][1], ak.corr(x, y))


def test_int64_overflow():
    np.random.seed(42)
    x = np.random.randint(2**61, 2**62, size=1000, dtype=np.int64)
    y = np.random.randint(2**61, 2**62, size=1000, dtype=np.int64)

    np.testing.assert_allclose(np.sum(x), ak.sum(x))
    np.testing.assert_allclose(np.mean(x), ak.mean(x))
    np.testing.assert_allclose(np.var(x), ak.var(x))
    np.testing.assert_allclose(np.std(x), ak.std(x))
    np.testing.assert_allclose(np.cov(x, y, ddof=0)[0][1], ak.covar(x, y))
    np.testing.assert_allclose(np.corrcoef(x, y)[0][1], ak.corr(x, y))


def test_complex_is_not_cast():
    x = np.array([1 + 2j, 3 + 4j, 5 - 1j])

    np.testing.assert_allclose(np.mean(x), ak.mean(x))
    assert ak.mean(ak.Array([[1 + 2j, 3 + 4j], [5 - 1j]]), axis=1).to_list() == [
        2 + 3j,
        5 - 1j,
    ]


def test_timedelta_is_not_cast():
    x = np.array([1, 2, 6], dtype="timedelta64[s]")

    assert ak.mean(x) == np.mean(x) == np.timedelta64(3, "s")


def test_datetime_is_not_cast():
    x = np.array([1, 2, 6], dtype="datetime64[s]")

    with pytest.raises(ValueError, match="cannot compute the sum"):
        ak.mean(x)


def test_floating_point_is_cast_only_where_numpy_casts_it():
    layout = ak.contents.NumpyArray(np.array([0.1, 0.25, 0.7], dtype=np.float32))

    # np.mean, np.var, and np.std compute in the dtype of the input
    assert ak._do.integers_to_float64(layout) is layout
    # np.cov and np.corrcoef compute in at least float64
    assert ak._do.real_numbers_to_float64(layout).dtype == np.dtype(np.float64)


def test_float32_covar_and_corr():
    x = np.array([0.1, 0.25, 0.7], dtype=np.float32)
    y = np.array([0.3, 0.2, 0.9], dtype=np.float32)

    np.testing.assert_allclose(np.cov(x, y, ddof=0)[0][1], ak.covar(x, y), rtol=1e-12)
    np.testing.assert_allclose(np.corrcoef(x, y)[0][1], ak.corr(x, y), rtol=1e-12)
