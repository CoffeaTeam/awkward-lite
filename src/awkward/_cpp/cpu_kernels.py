# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import ctypes
import importlib.resources as importlib_resources
import platform

from awkward._cpp import _kernel_signatures

if platform.system() == "Windows":
    name = "awkward-cpu-kernels.dll"
elif platform.system() == "Darwin":
    name = "libawkward-cpu-kernels.dylib"
else:
    name = "libawkward-cpu-kernels.so"

libpath_ref = importlib_resources.files(__package__) / "lib" / name
with importlib_resources.as_file(libpath_ref) as libpath:
    lib = ctypes.cdll.LoadLibrary(str(libpath))

kernel = _kernel_signatures.by_signature(lib)
