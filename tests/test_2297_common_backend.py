# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import pytest

import awkward as ak


def test_to_rdataframe():
    pytest.importorskip("ROOT")
    array = ak.Array([100, 200, 300.0], backend="typetracer")
    with pytest.raises(
        TypeError,
        match="from an nplike without known data to an nplike with known data",
    ):
        ak.to_rdataframe({"array": array})
