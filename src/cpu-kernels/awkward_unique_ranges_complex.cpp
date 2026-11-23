// BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_unique_ranges_complex.cpp", line)

#include "awkward/kernels.h"

// awkward_unique_ranges for complex numbers, each stored as a real and an
// imaginary part. Two numbers are equal if both parts are.
template <typename T>
ERROR awkward_unique_ranges_complex(
  T* __restrict__ toptr,
  const int64_t* __restrict__ fromoffsets,
  int64_t offsetslength,
  int64_t* __restrict__ tooffsets) {
  int64_t m = 0;
  for (int64_t i = 0;  i < offsetslength - 1;  i++) {
    tooffsets[i] = m;
    toptr[2 * m] = toptr[2 * fromoffsets[i]];
    toptr[2 * m + 1] = toptr[2 * fromoffsets[i] + 1];
    m++;
    // start after the first value, which is not equal to itself if NaN
    for (int64_t k = fromoffsets[i] + 1; k < fromoffsets[i + 1]; k++) {
      if (toptr[2 * (m - 1)] != toptr[2 * k] ||
          toptr[2 * (m - 1) + 1] != toptr[2 * k + 1]) {
        toptr[2 * m] = toptr[2 * k];
        toptr[2 * m + 1] = toptr[2 * k + 1];
        m++;
      }
    }
  }
  tooffsets[offsetslength - 1] = m;

  return success();
}

#define WRAPPER(FUNC, T) \
  ERROR FUNC(T* toptr, const int64_t* fromoffsets, int64_t offsetslength, int64_t* tooffsets) { \
    return awkward_unique_ranges_complex<T>(toptr, fromoffsets, offsetslength, tooffsets); \
  }

WRAPPER(awkward_unique_ranges_complex64, float)
WRAPPER(awkward_unique_ranges_complex128, double)
WRAPPER(awkward_unique_ranges_complex256, long double)
