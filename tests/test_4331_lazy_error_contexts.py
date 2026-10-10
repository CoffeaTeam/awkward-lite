# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak
from awkward._errors import OperationErrorContext, SlicingErrorContext
from awkward.errors import AxisError


def test_operation_note_still_reports_name_args_and_kwargs():
    with pytest.raises(AxisError) as excinfo:
        ak.num(ak.Array([1, 2, 3]), axis=1)

    (note,) = excinfo.value.__notes__
    assert "This error occurred while calling" in note
    assert "ak.num(" in note
    assert "<Array [1, 2, 3] type='3 * int64'>" in note
    assert "axis = 1" in note


def test_slicing_note_still_reports_array_and_slice():
    with pytest.raises(IndexError) as excinfo:
        ak.Array([[1, 2, 3], [], [4, 5]])[[0, 1, 2], [5, 5, 5]]

    (note,) = excinfo.value.__notes__
    assert "This error occurred while attempting to slice" in note
    assert "<Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>" in note
    assert "([0, 1, 2], [5, 5, 5])" in note


def test_lazy_path_does_not_format_when_nothing_is_raised(monkeypatch):
    calls = []
    format_args = OperationErrorContext._format_args

    def spy(self, arguments):
        calls.append(arguments)
        return format_args(self, arguments)

    monkeypatch.setattr(OperationErrorContext, "_format_args", spy)

    array = ak.Array([[1, 2, 3], [], [4, 5]])
    assert ak.num(array, axis=1).to_list() == [3, 0, 2]
    assert calls == []

    with pytest.raises(AxisError):
        ak.num(array, axis=2)
    assert len(calls) == 1


def test_contexts_stay_lazy():
    array = ak.Array([[1, 2, 3], [], [4, 5]])

    operation = OperationErrorContext("ak.num", (array,), {"axis": 1})
    slicing = SlicingErrorContext(array, slice(1, None))
    with operation:
        pass
    with slicing:
        pass

    assert "args" not in vars(operation)
    assert "kwargs" not in vars(operation)
    assert "array" not in vars(slicing)
    assert "where" not in vars(slicing)

    # ... but reading the note still formats them on demand
    assert "axis = 1" in operation.note
    assert "1:" in slicing.note
