// BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

#ifndef AWKWARD_NUMERIC_SORT_H_
#define AWKWARD_NUMERIC_SORT_H_

#include <cmath>
#include <type_traits>

namespace awkward::kernel {
  // Shared by value sort and argsort. Integers compare at their native width;
  // NaNs precede all other values in both directions, matching kernel behavior.
  // Equivalent values, including pairs of NaNs and signed zeros, compare false.
  template <typename T, bool Ascending>
  struct NumericOrder {
    [[nodiscard]] bool operator()(T left, T right) const noexcept {
      if constexpr (!std::is_integral_v<T>) {
        if (std::isnan(right)) {
          return false;
        }
        if (std::isnan(left)) {
          return true;
        }
      }
      if constexpr (Ascending) {
        return left < right;
      } else {
        return left > right;
      }
    }
  };

  // Complex numbers, each stored as a real and an imaginary part, compare by
  // real part and then by imaginary part, each in NumericOrder. So complex
  // numbers without an imaginary part sort like their real parts.
  template <typename T, bool Ascending>
  struct ComplexOrder {
    [[nodiscard]] bool operator()(const T* left, const T* right) const noexcept {
      const NumericOrder<T, Ascending> order{};
      return order(left[0], right[0]) ||
             (!order(right[0], left[0]) && order(left[1], right[1]));
    }
  };
}

#endif // AWKWARD_NUMERIC_SORT_H_
