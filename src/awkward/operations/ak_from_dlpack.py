# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


from awkward._connect.dlpack import DLPackDevice
from awkward._dispatch import high_level_function
from awkward._layout import from_arraylib, wrap_layout
from awkward._nplikes.numpy import Numpy

__all__ = ("from_dlpack",)

numpy = Numpy.instance()

# Devices whose memory NumPy can read directly
_HOST_ACCESSIBLE_DEVICES = frozenset(
    (
        DLPackDevice.CPU,
        DLPackDevice.CUDA_PINNED,
        DLPackDevice.ROCM_PINNED,
        DLPackDevice.CUDA_MANAGED,
    )
)


@high_level_function()
def from_dlpack(
    array,
    *,
    regulararray=False,
    highlevel=True,
    behavior=None,
    primitive_policy="error",
    attrs=None,
):
    """Converts a DLPack-aware array into an Awkward Array.

    If the data are in main memory, they are not copied: the buffer is shared
    through the DLPack protocol. Data on another device (e.g. a GPU) are
    copied to main memory, which requires NumPy 2.1 or later and a producer
    that supports DLPack's cross-device copies.

    The resulting layout may involve the following #ak.contents.Content types
    (only):

    * #ak.contents.NumpyArray
    * #ak.contents.RegularArray if `regulararray=True`.

    Args:
        array: The DLPack-supporting array to convert into an Awkward Array.
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
        An #ak.Array built from the given DLPack-aware array.
    """
    try:
        dlpack_info_func = array.__dlpack_device__
    except AttributeError as err:
        raise TypeError(
            f"Expected an object that implements the DLPack protocol, received {type(array)}"
        ) from err
    device_type, _device_id = dlpack_info_func()

    if device_type in _HOST_ACCESSIBLE_DEVICES:
        array = numpy.from_dlpack(array)
    else:
        import numpy as np  # noqa: TID251

        try:
            # ask the producer for a copy in main memory
            array = np.from_dlpack(array, device="cpu")
        except (TypeError, BufferError, RuntimeError) as err:
            raise TypeError(
                f"cannot copy a DLPack array on device {device_type!r} to main memory "
                "(this requires NumPy 2.1 or later and a producer that supports "
                "DLPack's cross-device copies); copy it to main memory first"
            ) from err

    return wrap_layout(
        from_arraylib(array, regulararray, False, primitive_policy=primitive_policy),
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )
