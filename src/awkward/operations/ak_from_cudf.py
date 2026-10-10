# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_cudf",)


@high_level_function()
def from_cudf(obj, *, highlevel=True, behavior=None, attrs=None):
    """Converts a cuDF Series or DataFrame into an Awkward Array.

    cuDF copies the data from the GPU into main memory as Arrow, and the Arrow
    data are converted as in #ak.from_arrow. A DataFrame becomes an array of
    records with one field per column, and its index is dropped. Categorical
    data become categorical Awkward Arrays.

    This function requires the `cudf` library and a compatible GPU.

    See also #ak.to_cudf, #ak.from_cupy, #ak.from_arrow.

    Args:
        obj (cudf.Series or cudf.DataFrame): The cuDF object to convert into
            an Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns:
        An #ak.Array with the same data as `obj`.
    """
    # Dispatch
    yield (obj,)

    # Implementation
    return _impl(obj, highlevel, behavior, attrs)


def _impl(obj, highlevel, behavior, attrs):
    try:
        import cudf
    except ImportError as err:
        raise ImportError(
            """to use ak.from_cudf, you must install the 'cudf' package with:

    pip install cudf-cu13
or
    conda install -c rapidsai cudf cuda-version=13"""
        ) from err

    if isinstance(obj, cudf.DataFrame):
        arrow_data = obj.to_arrow(preserve_index=False)
    elif isinstance(obj, cudf.Series):
        arrow_data = obj.to_arrow()
    else:
        raise TypeError(
            "ak.from_cudf accepts only cudf.Series or cudf.DataFrame, "
            f"not {type(obj).__name__!r}"
        )

    return ak.operations.ak_from_arrow._impl(
        arrow_data,
        generate_bitmasks=False,
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )
