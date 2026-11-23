# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

import pytest

import awkward as ak


@pytest.mark.parametrize("forget_length", [False, True])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_argsort_typetracer_of_lists_with_missing_values(forget_length, axis):
    array = ak.Array([[[1, None], [2]], [[3]], [None, [4, 5]]])
    tracer = ak.Array(array.layout.to_typetracer(forget_length=forget_length))

    result = ak.argsort(tracer, axis=axis)
    assert result.type.content == ak.argsort(array, axis=axis).type.content
