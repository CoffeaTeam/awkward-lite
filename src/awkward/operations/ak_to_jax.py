# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_jax",)


@high_level_function()
def to_jax(array):
    """Converts an Awkward Array into a JAX Array, if possible.

    If the data are numerical and regular (nested lists have equal lengths in
    each dimension, as described by the #ak.Array.type), they can be losslessly
    converted to a JAX array and this function returns without an error.

    Otherwise, the function raises an error.

    The JAX Array is placed on JAX's default device, and its dtype follows
    JAX's own rules (e.g. 64-bit data become 32-bit unless the `jax_enable_x64`
    option is set).

    See also #ak.from_jax and #ak.to_numpy.

    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    Returns:
        A JAX array with the same data as `array`, if the conversion is possible.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    try:
        import jax
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """to use ak.to_jax, you must install the 'jax' package with:

    pip install jax

or

    conda install -c conda-forge jax"""
        ) from err

    numpy_array = ak.operations.ak_to_numpy._impl(array, allow_missing=False)

    return jax.numpy.asarray(numpy_array)
