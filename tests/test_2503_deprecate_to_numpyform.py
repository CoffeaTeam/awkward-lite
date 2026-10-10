# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import awkward as ak


def test_primitive():
    form = ak.forms.EmptyForm()
    next_form = form.to_NumpyForm(primitive="int64")
    assert next_form.primitive == "int64"
