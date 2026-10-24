// BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

// The form of a builder of long double values names the primitive as NumPy
// does, by its size. The size depends on the platform.

#include "awkward/LayoutBuilder.h"

#include <cassert>
#include <complex>
#include <cstdlib>
#include <map>
#include <string>

template <class PRIMITIVE>
using NumpyBuilder = awkward::LayoutBuilder::Numpy<PRIMITIVE>;

void
test_type_to_name() {
  assert(awkward::type_to_name<long double>() ==
         "float" + std::to_string(8 * sizeof(long double)));
  assert(awkward::type_to_name<std::complex<long double>>() ==
         "complex" + std::to_string(16 * sizeof(long double)));

  // the other names do not change
  assert(awkward::type_to_name<float>() == "float32");
  assert(awkward::type_to_name<double>() == "float64");
  assert(awkward::type_to_name<std::complex<double>>() == "complex128");
}

void
test_Numpy_long_double() {
  NumpyBuilder<long double> builder;
  builder.append(1.1L);
  builder.append(2.2L);
  assert(builder.length() == 2);

  std::string error;
  assert(builder.is_valid(error) == true);

  assert(builder.form() ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"" + awkward::type_to_name<long double>() + "\", "
      "\"form_key\": \"node0\" "
  "}");

  std::map<std::string, size_t> names_nbytes = {};
  builder.buffer_nbytes(names_nbytes);
  assert(names_nbytes["node0-data"] == 2 * sizeof(long double));

  std::map<std::string, void*> buffers = {};
  buffers["node0-data"] = malloc(names_nbytes["node0-data"]);
  builder.to_buffers(buffers);

  long double* data = (long double*)buffers["node0-data"];
  assert(data[0] == 1.1L);
  assert(data[1] == 2.2L);

  free(buffers["node0-data"]);
}

void
test_Numpy_complex_long_double() {
  NumpyBuilder<std::complex<long double>> builder;
  builder.append({1.1L, 0.1L});
  assert(builder.length() == 1);

  assert(builder.form() ==
  "{ "
      "\"class\": \"NumpyArray\", "
      "\"primitive\": \"" + awkward::type_to_name<std::complex<long double>>() + "\", "
      "\"form_key\": \"node0\" "
  "}");
}

int
main(int /* argc */, char** /* argv */) {
  test_type_to_name();
  test_Numpy_long_double();
  test_Numpy_complex_long_double();
  return 0;
}
