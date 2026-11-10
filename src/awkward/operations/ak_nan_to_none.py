# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import awkward as ak
from awkward._dispatch import high_level_function
from awkward._layout import HighLevelContext
from awkward._nplikes.numpy_like import NumpyMetadata

__all__ = ("nan_to_none",)

np = NumpyMetadata.instance()


@high_level_function()
def nan_to_none(
    array,
    *,
    highlevel=True,
    behavior=None,
    attrs=None,
):
    """Converts NaN ("not a number") into None, i.e. missing values with option-type.

    A complex number is NaN if either of its parts is, as in `np.isnan`.

    See also #ak.nan_to_num to convert NaN or infinity to specified values.

    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns:
        An array with NaN ("not a number") converted to None, i.e. missing
        values with option-type.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, highlevel, behavior, attrs)


def _impl(array, highlevel, behavior, attrs):
    def action(layout, continuation, backend, **kwargs):
        if layout.is_numpy and layout.dtype.kind in "fc":
            mask = backend.nplike.isnan(layout.data)
            return ak.contents.ByteMaskedArray(
                ak.index.Index8(mask, nplike=backend.nplike),
                layout,
                valid_when=False,
            )

        elif (layout.is_option or layout.is_indexed) and (
            layout.content.is_numpy and layout.content.dtype.kind in "fc"
        ):
            return continuation()

        else:
            return None

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False, primitive_policy="error")

    out = ak._do.recursively_apply(layout, action)
    return ctx.wrap(out, highlevel=highlevel)
