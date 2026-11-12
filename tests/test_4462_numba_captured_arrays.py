# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

from __future__ import annotations

import gc
import os
import subprocess
import sys
import weakref

import pytest

import awkward as ak

numba = pytest.importorskip("numba")
ak.numba.register_and_check()


@pytest.mark.parametrize("returned_view", [False, True])
@pytest.mark.parametrize("mutation", ["field", "behavior"])
def test_captured_lookup_survives_array_update(returned_view, mutation):
    events = ak.Array([{"x": [1.5, 2.5, 3.5]}, {"x": [4.5]}] * 10)
    if returned_view:

        @numba.njit
        def read():
            return events[0].x

        assert read().to_list() == [1.5, 2.5, 3.5]
    else:

        @numba.njit
        def read():
            total = 0.0
            for event in events:
                for value in event.x:
                    total += value
            return total

        assert read() == 120.0

    owner = weakref.ref(events._numbaview.lookup)
    if mutation == "field":
        events["y"] = events.x * 2
    else:
        events.behavior = {}
    gc.collect()
    # Fail safely before accessing a dangling pointer on the old implementation.
    assert owner() is not None
    junk = [bytes([0x41 + i % 3]) * k for i in range(1000) for k in range(8, 24)]
    for _ in range(3):
        if returned_view:
            assert read().to_list() == [1.5, 2.5, 3.5]
        else:
            assert read() == 120.0
    assert junk


@pytest.mark.skipif(sys.platform == "emscripten", reason="requires fresh subprocesses")
@pytest.mark.thread_unsafe
def test_disk_cache_with_global_and_argument_arrays(tmp_path):
    script = tmp_path / "cached_arrays.py"
    script.write_text(
        """
import sys
import warnings

import awkward as ak
import numba
from numba.core.errors import NumbaWarning

ak.numba.register_and_check()
value, expected_hits = map(int, sys.argv[1:])
array = ak.Array([value, value + 1])

@numba.njit(cache=True)
def global_read(index):
    return array[index]

@numba.njit(cache=True)
def argument_read(array, index):
    return array[index]

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", NumbaWarning)
    assert global_read(1) == value + 1
assert any(
    issubclass(w.category, NumbaWarning)
    and "dynamic globals" in str(w.message)
    for w in caught
), caught

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", NumbaWarning)
    assert argument_read(array, 1) == value + 1
assert not any(
    issubclass(w.category, NumbaWarning) and "dynamic globals" in str(w.message)
    for w in caught
), caught
assert sum(argument_read.stats.cache_hits.values()) == expected_hits
assert sum(argument_read.stats.cache_misses.values()) == 1 - expected_hits
"""
    )
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "cache")}
    # Both processes use the same source and disk cache but different buffers
    # and values. Global pointers must be rebuilt; argument code must be reused.
    for value, hits in [(17, 0), (29, 1)]:
        result = subprocess.run(
            [sys.executable, str(script), str(value), str(hits)],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
