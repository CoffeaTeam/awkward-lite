# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import contextlib
import datetime
import io
import os
import time

import yaml

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@contextlib.contextmanager
def write_if_changed(path):
    """Yield a buffer to write to; only touch the file if the content changed
    (ignoring the generation timestamp), so unchanged outputs keep their
    mtimes and don't trigger rebuilds."""
    buffer = io.StringIO()
    yield buffer
    content = buffer.getvalue()

    def strip_stamp(text):
        return [line for line in text.splitlines() if "AUTO GENERATED ON" not in line]

    if os.path.exists(path):
        with open(path) as file:
            if strip_stamp(file.read()) == strip_stamp(content):
                return
    with open(path, "w") as file:
        file.write(content)


def reproducible_datetime():
    timestamp = int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
    build_date = datetime.datetime.fromtimestamp(timestamp, tz=datetime.UTC)
    return build_date.isoformat().replace("T", " AT ")[:22]


def type_to_ctype(typename):
    is_const = False
    if "Const[" in typename:
        is_const = True
        typename = typename[len("Const[") : -1]
    count = 0
    while "List[" in typename:
        count += 1
        typename = typename[len("List[") : -1]
    typename = typename + "*" * count
    if is_const:
        typename = "const " + typename
    return typename


def include_kernels_h(specification):
    print("Generating awkward-cpp/include/awkward/kernels.h...")

    with write_if_changed(
        os.path.join(
            CURRENT_DIR, "..", "awkward-cpp", "include", "awkward", "kernels.h"
        ),
    ) as header:
        header.write(
            f"""// AUTO GENERATED ON {reproducible_datetime()}
// DO NOT EDIT BY HAND!
//
// To regenerate file, run
//
//     python dev/generate-kernel-signatures.py
//
// (It is usually run as part of pip install . or localbuild.py.)

#ifndef AWKWARD_KERNELS_H_
#define AWKWARD_KERNELS_H_

#include "awkward/common.h"

extern "C" {{

"""
        )
        for spec in specification["kernels"]:
            for childfunc in spec["specializations"]:
                header.write(" " * 2 + "EXPORT_SYMBOL ERROR\n")
                header.write(" " * 2 + childfunc["name"] + "(\n")
                for i, arg in enumerate(childfunc["args"]):
                    header.write(
                        " " * 4 + type_to_ctype(arg["type"]) + " " + arg["name"]
                    )
                    if i == (len(childfunc["args"]) - 1):
                        header.write(");\n")
                    else:
                        header.write(",\n")
            header.write("\n")
        header.write(
            """}

#endif // AWKWARD_KERNELS_H_
"""
        )

    print("           awkward-cpp/include/awkward/kernels.h.")


type_to_dtype = {
    "bool": "bool_",
    "int8": "int8",
    "uint8": "uint8",
    "int16": "int16",
    "uint16": "uint16",
    "int32": "int32",
    "uint32": "uint32",
    "int64": "int64",
    "uint64": "uint64",
    "float": "float32",
    "double": "float64",
}


def type_to_pytype(typename, special):
    if "Const[" in typename:
        typename = typename[len("Const[") : -1]
    count = 0
    while "List[" in typename:
        count += 1
        typename = typename[len("List[") : -1]
    if typename.endswith("_t"):
        typename = typename[:-2]
    if count != 0:
        special.append(type_to_dtype[typename])
    return ("POINTER(" * count) + ("c_" + typename) + (")" * count)


def kernel_signatures_py(specification):
    print("Generating awkward-cpp/src/awkward_cpp/_kernel_signatures.py...")

    with write_if_changed(
        os.path.join(
            CURRENT_DIR,
            "..",
            "awkward-cpp",
            "src",
            "awkward_cpp",
            "_kernel_signatures.py",
        ),
    ) as file:
        file.write(
            f"""# AUTO GENERATED ON {reproducible_datetime()}
# DO NOT EDIT BY HAND!
#
# To regenerate file, run
#
#     python dev/generate-kernel-signatures.py
#
# (It is usually run as part of pip install . or localbuild.py.)

# fmt: off

from ctypes import (
    POINTER,
    Structure,
    c_bool,
    c_int8,
    c_uint8,
    c_int16,
    c_uint16,
    c_int32,
    c_uint32,
    c_int64,
    c_uint64,
    c_float,
    c_double,
    c_char_p,
)

import numpy as np

from numpy import (
    bool_,
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float32,
    float64,
)

class ERROR(Structure):
    _fields_ = [
        ("str", c_char_p),
        ("filename", c_char_p),
        ("id", c_int64),
        ("attempt", c_int64),
    ]


def by_signature(lib):
    out = {{}}
"""
        )

        for spec in specification["kernels"]:
            for childfunc in spec["specializations"]:
                special = [repr(spec["name"])]
                arglist = [
                    type_to_pytype(x["type"], special) for x in childfunc["args"]
                ]
                dirlist = [repr(x["dir"]) for x in childfunc["args"]]
                file.write(
                    """
    f = lib.{}
    f.argtypes = [{}]
    f.restype = ERROR
    f.dir = [{}]
    out[{}] = f
""".format(
                        str(childfunc["name"]),
                        ", ".join(arglist),
                        ", ".join(dirlist),
                        ", ".join(special),
                    )
                )

        file.write(
            """
    return out
"""
        )

    print("           awkward-cpp/src/awkward_cpp/_kernel_signatures.py...")


if __name__ == "__main__":
    with open(os.path.join(CURRENT_DIR, "..", "kernel-specification.yml")) as specfile:
        specification = yaml.safe_load(specfile)
    include_kernels_h(specification)
    kernel_signatures_py(specification)
