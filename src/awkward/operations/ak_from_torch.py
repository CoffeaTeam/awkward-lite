# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import awkward as ak
from awkward._dispatch import high_level_function

__all__ = ("from_torch",)


@high_level_function()
def from_torch(array):
    """Converts a PyTorch Tensor into an Awkward Array.

    A CPU tensor shares its buffer with the Awkward Array (the data are not
    copied); a tensor on another device (e.g. a GPU) is copied to main memory.

    If `array` contains any other data types the function raises an error.

    Args:
        array: (PyTorch Tensor):
            Tensor to convert into an Awkward Array.

    Returns:
        An #ak.Array built from the given PyTorch tensor.
    """

    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array)


def _impl(array):
    try:
        import torch
    except ImportError as err:
        raise ImportError(
            """to use ak.from_torch, you must install 'torch' package with:

         pip install torch

or

        conda install pytorch"""
        ) from err

    # check if array is a Tensor
    if not isinstance(array, torch.Tensor):
        raise TypeError("""only PyTorch Tensor can be converted to Awkward Array""")

    # `Tensor.cpu` is a no-op for CPU tensors and copies tensors on other
    # devices (e.g. GPUs) into main memory
    return ak.from_numpy(array.cpu().numpy())
