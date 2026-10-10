# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_cupy",)


@high_level_function()
def to_cupy(array):
    """Converts an Awkward Array into a CuPy array, if possible.

    If the data are numerical and regular (nested lists have equal lengths in
    each dimension, as described by the #ak.Array.type), they can be losslessly
    converted to a CuPy array and this function returns without an error.

    Otherwise, the function raises an error.

    The data are copied from main memory to the current CuPy device.

    See also #ak.from_cupy and #ak.to_numpy.

    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    Returns:
        A CuPy array with the same data as `array`, if the conversion is possible.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    try:
        import cupy
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """to use ak.to_cupy, you must install the 'cupy' package with:

    pip install cupy

or

    conda install -c conda-forge cupy"""
        ) from err

    numpy_array = ak.operations.ak_to_numpy._impl(array, allow_missing=False)

    return cupy.asarray(numpy_array)
