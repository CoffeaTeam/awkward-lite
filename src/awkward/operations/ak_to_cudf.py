# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_cudf",)


@high_level_function()
def to_cudf(array):
    """Converts an Awkward Array into a cuDF Series.

    The array is converted to Arrow in main memory (see #ak.to_arrow), and
    cuDF copies it to the GPU. Regular dimensions become variable-length
    lists, and categorical data become cuDF categoricals. Types that cuDF
    cannot represent, such as unions, raise an error.

    This function requires the `cudf` library and a compatible GPU.

    See also #ak.to_cupy, #ak.to_arrow, #ak.to_dataframe.

    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    Returns:
        A cuDF Series with the same data as `array`.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    try:
        import cudf
    except ImportError as err:
        raise ImportError(
            """to use ak.to_cudf, you must install the 'cudf' package with:

    pip install cudf-cu13
or
    conda install -c rapidsai cudf cuda-version=13"""
        ) from err

    layout = ak.to_layout(array, allow_record=False)

    # cuDF has no fixed-size lists
    layout = ak.operations.ak_from_regular._impl(
        layout, axis=None, highlevel=False, behavior=None, attrs=None
    )

    arrow_array = ak.operations.ak_to_arrow._impl(
        layout,
        # cuDF lists and strings have 32-bit offsets
        list_to32=True,
        string_to32=True,
        bytestring_to32=True,
        emptyarray_to="float64",
        categorical_as_dictionary=True,
        extensionarray=False,
        count_nulls=True,
    )
    return cudf.Series.from_arrow(arrow_array)
