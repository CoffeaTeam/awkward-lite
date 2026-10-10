# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


from awkward._dispatch import high_level_function
from awkward._layout import from_arraylib, wrap_layout
from awkward._nplikes.numpy import Numpy

__all__ = ("from_jax",)

numpy = Numpy.instance()


@high_level_function()
def from_jax(
    array,
    *,
    regulararray=False,
    highlevel=True,
    behavior=None,
    attrs=None,
    primitive_policy="error",
):
    """Converts a JAX Array into an Awkward Array.

    The data are brought into main memory: if the JAX Array is on the CPU, the
    Awkward Array shares its (read-only) buffer; otherwise, the data are copied
    from the device.

    The resulting layout may involve the following #ak.contents.Content types
    (only):

    * #ak.contents.NumpyArray
    * #ak.contents.RegularArray if `regulararray=True`.

    See also #ak.to_jax, #ak.from_numpy and #ak.from_cupy.

    Args:
        array (jax.Array): The JAX Array to convert into an Awkward Array.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.contents.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.contents.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        attrs (None or dict): Custom attributes for the output array, if
            high-level.

    Returns:
        An #ak.Array built from the given JAX array.
    """
    try:
        import jax
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            """to use ak.from_jax, you must install the 'jax' package with:

    pip install jax

or

    conda install -c conda-forge jax"""
        ) from err

    if not isinstance(array, jax.Array):
        raise TypeError(
            f"only JAX Arrays can be converted by ak.from_jax, not {type(array).__name__}"
        )

    return wrap_layout(
        from_arraylib(
            numpy.asarray(array), regulararray, False, primitive_policy=primitive_policy
        ),
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )
