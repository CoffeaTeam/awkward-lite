# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_tensorflow",)


@high_level_function()
def from_tensorflow(array):
    """Converts a TensorFlow Tensor into an Awkward Array.

    The tensor is copied into main memory (from its device, if need be),
    because a NumPy array is mutable and a TensorFlow tensor is not.

    If `array` contains any other data types the function raises an error.

    Args:
        array: (TensorFlow Tensor):
            Tensor to convert into an Awkward Array.

    Returns:
        An #ak.Array built from the given TensorFlow Tensor.
    """

    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    try:
        import tensorflow as tf
    except ImportError as err:
        raise ImportError(
            """to use ak.from_tensorflow, you must install the 'tensorflow' package with:

        pip install tensorflow
or
        conda install tensorflow"""
        ) from err

    # check if array is a Tensor
    if not isinstance(array, tf.Tensor):
        raise TypeError(
            """only a TensorFlow Tensor can be converted to Awkward Array"""
        )

    # this makes a copy (from the tensor's device, if need be), since a NumPy
    # array is mutable and a TensorFlow tensor is not
    return ak.from_numpy(array.numpy())
