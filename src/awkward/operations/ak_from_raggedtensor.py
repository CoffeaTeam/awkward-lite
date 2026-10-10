# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_raggedtensor",)


@high_level_function()
def from_raggedtensor(array):
    """Converts a TensorFlow RaggedTensor into an Awkward Array.

    The underlying buffers (flat values and row splits) are copied into main
    memory (from the RaggedTensor's device, if need be).

    If `array` contains any other data types the function raises an error.

    Args:
        array: (`tensorflow.RaggedTensor`):
            RaggedTensor to convert into an  Awkward Array.

    Returns:
        An #ak.Array built from the given TensorFlow RaggedTensor.
    """

    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    try:
        # get the flat values
        content = array.flat_values
    except AttributeError as err:
        raise TypeError(
            """only RaggedTensor can be converted to awkward array"""
        ) from err

    # convert flat_values to ak.contents right away
    content = ak.contents.NumpyArray(content.numpy())

    # get the offsets
    offsets_arr = []
    for splits in array.nested_row_splits:
        # convert to ak.index
        offset = ak.index.Index64(splits.numpy())
        offsets_arr.append(offset)

    # if a tensor has one *ragged dimension*
    if len(offsets_arr) == 1:
        result = ak.contents.ListOffsetArray(offsets_arr[0], content)
        return ak.Array(result)

    # if a tensor has multiple *ragged dimensions*
    return ak.Array(_recursive_call(content, offsets_arr, 0))


def _recursive_call(content, offsets_arr, count):
    if count == len(offsets_arr) - 2:
        return ak.contents.ListOffsetArray(
            offsets_arr[count],
            ak.contents.ListOffsetArray(offsets_arr[count + 1], content),
        )
    else:
        return ak.contents.ListOffsetArray(
            offsets_arr[count], _recursive_call(content, offsets_arr, count + 1)
        )
