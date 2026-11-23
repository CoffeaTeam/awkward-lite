// BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_sort_complex.cpp", line)

#include "awkward/kernels.h"
#include "awkward/numeric-sort.h"

#include <algorithm>
#include <numeric>
#include <vector>

// Per-segment sort of complex numbers in ComplexOrder. (NumPy orders them in
// the same way, but sorts NaN to the high end.)
template <typename T>
ERROR awkward_sort_complex(
  T* __restrict__ toptr,
  const T* __restrict__ fromptr,
  int64_t length,
  const int64_t* __restrict__ offsets,
  int64_t offsetslength,
  bool ascending,
  bool stable) {
  std::vector<int64_t> index(length);
  std::iota(index.begin(), index.end(), int64_t{0});

  auto order_ascending = [fromptr](int64_t i1, int64_t i2) {
    return awkward::kernel::ComplexOrder<T, true>{}(fromptr + 2 * i1, fromptr + 2 * i2);
  };
  auto order_descending = [fromptr](int64_t i1, int64_t i2) {
    return awkward::kernel::ComplexOrder<T, false>{}(fromptr + 2 * i1, fromptr + 2 * i2);
  };

  // Each segment of `index` is sorted independently.
  for (int64_t i = 0; i < offsetslength - 1; i++) {
    // Clamp against `length` so malformed offsets cannot index past the end
    // of `index` (well-formed offsets satisfy offsets[last] <= length).
    int64_t start_off = static_cast<int64_t>(offsets[i]);
    int64_t stop_off = static_cast<int64_t>(offsets[i + 1]);
    if (start_off > length) start_off = length;
    if (stop_off > length) stop_off = length;
    if (start_off >= stop_off) continue;
    auto start = index.begin() + start_off;
    auto stop = index.begin() + stop_off;
    if (stable) {
      if (ascending) {
        std::stable_sort(start, stop, order_ascending);
      } else {
        std::stable_sort(start, stop, order_descending);
      }
    } else {
      if (ascending) {
        std::sort(start, stop, order_ascending);
      } else {
        std::sort(start, stop, order_descending);
      }
    }
  }

  // In the offsets representation, the number of elements to copy out is
  // simply the end of the last bin (offsets[0] == 0 by contract).
  int64_t total = (offsetslength > 0) ? offsets[offsetslength - 1] : 0;
  int64_t copy_length = (total < length) ? total : length;
  for (int64_t i = 0; i < copy_length; i++) {
    toptr[2 * i] = fromptr[2 * index[i]];
    toptr[2 * i + 1] = fromptr[2 * index[i] + 1];
  }
  return success();
}

#define SORT(FUNC, T)                                                      \
  ERROR FUNC(                                                              \
    T* toptr, const T* fromptr, int64_t length,                            \
    const int64_t* offsets, int64_t offsetslength,                         \
    bool ascending, bool stable) {                                         \
    return awkward_sort_complex<T>(                                        \
      toptr, fromptr, length, offsets, offsetslength,                      \
      ascending, stable);                                                  \
  }

SORT(awkward_sort_complex64, float)
SORT(awkward_sort_complex128, double)
SORT(awkward_sort_complex256, long double)
