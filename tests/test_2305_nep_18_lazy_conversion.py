# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak


def test_binary():
    # awkward expects native byteorder
    dtype = np.dtype("u4")
    ak_array = ak.Array(np.arange(10, dtype=dtype))
    np_array = np.arange(10, dtype=dtype.newbyteorder("S"))
    with pytest.raises(TypeError):
        # ak.array_equal now overrides np.array_equal, and requires
        # both arrays to be valid within awkward.
        assert np.array_equal(ak_array, np_array)
