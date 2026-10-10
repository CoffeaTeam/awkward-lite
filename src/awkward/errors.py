# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE


import numpy

__all__ = ("AxisError", "FieldNotFoundError")


class FieldNotFoundError(IndexError):
    pass


AxisError = numpy.exceptions.AxisError
