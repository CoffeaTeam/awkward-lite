# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("to_tensorflow",)


@high_level_function()
def to_tensorflow(array):
    """Converts an Awkward Array into a TensorFlow Tensor, if possible.

    If `array` contains any other data types (RecordArray for example) the
    function raises a TypeError.

    Args:
        array: Array-like data (anything #ak.to_layout recognizes).

    Returns:
        A TensorFlow Tensor with the same data as `array`, if the conversion is
        possible.
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
            """to use ak.to_tensorflow, you must install the 'tensorflow' package with:

        pip install tensorflow
or
        conda install tensorflow"""
        ) from err

    # useful function that handles all possible input arrays
    array = ak.to_layout(array, allow_record=False)

    if ak.backend(array) != "cpu":
        raise ValueError("""Only 'cpu' backend conversions are allowed""")

    try:
        backend_array = array.to_backend_array(allow_missing=False)
    except ValueError as err:
        raise TypeError(
            "Only arrays containing equal-length lists of numbers can be converted into a TensorFlow Tensor"
        ) from err

    with tf.device("CPU:0"):
        return tf.convert_to_tensor(backend_array, dtype=tf.float64)
