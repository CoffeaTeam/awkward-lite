// BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_argsort_complex.cpp", line)

#include <algorithm>
#include <numeric>
#include <vector>

#include "awkward/kernels.h"
#include "awkward/numeric-sort.h"

// Per-segment argsort of complex numbers in ComplexOrder, as in
// awkward_sort_complex.
template <typename T, typename U>
ERROR awkward_argsort_complex(
  int64_t* __restrict__ toptr,
  const T* __restrict__ fromptr,
  int64_t length,
  const U* __restrict__ offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  std::iota(toptr, toptr + length, int64_t{0});

  auto order_ascending = [fromptr](int64_t i1, int64_t i2) {
    return awkward::kernel::ComplexOrder<T, true>{}(fromptr + 2 * i1, fromptr + 2 * i2);
  };
  auto order_descending = [fromptr](int64_t i1, int64_t i2) {
    return awkward::kernel::ComplexOrder<T, false>{}(fromptr + 2 * i1, fromptr + 2 * i2);
  };

  // Each segment of `toptr` is sorted independently.
  for (int64_t i = 0; i < offsetslength - 1; i++) {
    // Clamp against `length` so malformed offsets cannot index past the end
    // of `toptr` (well-formed offsets satisfy offsets[last] <= length).
    int64_t start_off = static_cast<int64_t>(offsets[i]);
    int64_t stop_off = static_cast<int64_t>(offsets[i + 1]);
    if (start_off > length) start_off = length;
    if (stop_off > length) stop_off = length;
    if (start_off >= stop_off) continue;
    int64_t* segment_start = toptr + start_off;
    int64_t* segment_stop = toptr + stop_off;
    if (stable) {
      if (ascending) {
        std::stable_sort(segment_start, segment_stop, order_ascending);
      } else {
        std::stable_sort(segment_start, segment_stop, order_descending);
      }
    } else {
      if (ascending) {
        std::sort(segment_start, segment_stop, order_ascending);
      } else {
        std::sort(segment_start, segment_stop, order_descending);
      }
    }
    std::transform(segment_start, segment_stop, segment_start,
                   [start_off](int64_t j) { return j - start_off; });
  }
  return success();
}

#define ARGSORT_OFF(FUNC, T, U)                                            \
  ERROR FUNC(                                                              \
    int64_t* toptr, const T* fromptr, int64_t length,                      \
    const U* offsets, int64_t offsetslength,                               \
    bool ascending, bool stable) {                                         \
    return awkward_argsort_complex<T, U>(                                  \
      toptr, fromptr, length, offsets, offsetslength, ascending, stable);  \
  }

ARGSORT_OFF(awkward_argsort_complex64_int32, float, int32_t)
ARGSORT_OFF(awkward_argsort_complex64_uint32, float, uint32_t)
ARGSORT_OFF(awkward_argsort_complex64_uint64, float, uint64_t)
ARGSORT_OFF(awkward_argsort_complex64_int64, float, int64_t)

ARGSORT_OFF(awkward_argsort_complex128_int32, double, int32_t)
ARGSORT_OFF(awkward_argsort_complex128_uint32, double, uint32_t)
ARGSORT_OFF(awkward_argsort_complex128_uint64, double, uint64_t)
ARGSORT_OFF(awkward_argsort_complex128_int64, double, int64_t)

ARGSORT_OFF(awkward_argsort_complex256_int32, long double, int32_t)
ARGSORT_OFF(awkward_argsort_complex256_uint32, long double, uint32_t)
ARGSORT_OFF(awkward_argsort_complex256_uint64, long double, uint64_t)
ARGSORT_OFF(awkward_argsort_complex256_int64, long double, int64_t)
