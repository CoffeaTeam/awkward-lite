# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Repository layout: one package

This repo is a **single package**, `awkward`, built with scikit-build-core (`pyproject.toml` and `CMakeLists.txt` at the root). It is installed from the repository, not from PyPI. Everything installs inside the `awkward` package:

- **`awkward`** — the Python code, in `src/awkward`.
- **`awkward._cpp`** — the compiled C++ components: CMake installs the libraries (the pybind11 module `_ext`, `libawkward` and the CPU kernels) into `awkward/_cpp/lib`, and `src/awkward/_cpp` loads them. The C++ sources are in `src/cpu-kernels`, `src/libawkward` and `src/python`, the headers in `include/awkward`, and RapidJSON is a submodule in `extern/rapidjson`.

Upstream awkward ships the C++ part as a separate `awkward-cpp` package, imported as `awkward_cpp`. Here its files are at the root (`awkward-cpp/X` → `X`, except `awkward-cpp/src/awkward_cpp` → `src/awkward/_cpp`, `awkward-cpp/rapidjson` → `extern/rapidjson` and `awkward-cpp/docs` → `docs/doxygen`), and `awkward_cpp` is imported as `awkward._cpp`.

Parts of the C++ build are generated; building from the repository generates them automatically (CMake runs the `dev/` generation scripts at configure time).

## Development setup

```bash
git clone --recursive https://github.com/CoffeaTeam/awkward-lite.git  # rapidjson submodule
uv sync   # builds the C++ code (generating headers, kernel signatures, and kernel tests) and installs the package editable
```

Any `uv run ...` command does the same setup on demand and rebuilds the C++ code when its sources or the kernel spec change (see `cache-keys` in `pyproject.toml`). The `dev` dependency group holds the test dependencies. Without uv:

```bash
python -m pip install -v -e .
```

- `nox -s prepare` runs the generation scripts without building; it accepts `--headers --signatures --tests --docs` to select targets (`--docs` data is the only target the build doesn't cover). `noxfile.py` is a uv script (`./noxfile.py -s prepare` also works without nox installed); `nox -R -s ...` reuses the session venv.
- Python-only changes need no rebuild (editable install). Rebuilding is only needed when C++ or the kernel spec changes; the CMake build directory is kept in `build/`, so rebuilds are incremental. For fast iteration install the build deps (including `pyyaml` and `numpy`) and use `pip install --no-build-isolation --check-build-dependencies -e .`.

## Testing

```bash
python -m pytest -n auto tests                      # main test suite
python -m pytest tests/test_XXXX-description.py     # single test file
python -m pytest -n auto tests-spec         # generated kernel-spec tests (Python reference impls)
python -m pytest -n auto tests-cpu-kernels  # generated tests against compiled kernels
```

The generated kernel test dirs are created by building the package from the repository (or by `nox -s prepare -- --tests`).

Test files are named `tests/test_XXXX-description.py` where `XXXX` is the GitHub issue or PR number (enforced by `dev/validate-test-names.py`). New code needs a new test file following this convention.

Warnings are errors in pytest (`filterwarnings = ["error", ...]`).

## Linting

```bash
prek -a --quiet    # or: nox -s lint (or pre-commit run -a if prek is missing)
```

There is also `nox -s pylint`.

## Architecture

### Layered Python design (`src/awkward`)

- `highlevel.py` — user-facing `ak.Array`, `ak.Record`, `ak.ArrayBuilder`. These wrap a low-level layout.
- `operations/ak_*.py` — one file per public `ak.*` function. Each goes through `_dispatch.py` (`@high_level_function`), which enables third-party overload dispatch; the function body is split into a `dispatch` generator yielding array arguments, then the `_impl` function.
- `contents/` — the layout node types (`NumpyArray`, `ListOffsetArray`, `RecordArray`, `IndexedOptionArray`, `UnionArray`, etc.), forming a tree that represents nested/ragged data columnar-wise. `index.py` holds the integer index buffers. `_do.py` contains cross-cutting operations on layouts.
- `forms/` — metadata-only mirrors of each content type (the "form" = type + buffer structure without data), used for serialization and typetracer.
- `_meta/` — shared base logic between contents and forms.
- `_backends/` and `_nplikes/` — abstraction over array libraries: NumPy and **typetracer** (shape-only arrays with possibly-unknown lengths, used by dask-awkward to compute without data). Code in `contents/` must go through the nplike API, not call NumPy directly.
- `_kernels.py` + `_cpp/_kernel_signatures.py` — how layouts invoke compiled kernels via ctypes-style signatures, per backend.
- `_connect/` — integrations: numba, pyarrow (Arrow/Parquet), numexpr, RDataFrame, cling/cppyy.
- `behaviors/` — built-in behaviors (e.g. strings as character lists) layered on the `ak.behavior` registry, which maps record names/parameters to Python mixin classes.
- `_broadcasting.py`, `_slicing.py`, `_reducers.py` — the core algorithms behind ufuncs, `__getitem__`, and reductions.

### Kernels are spec-driven

`kernel-specification.yml` (repo root) is the source of truth for every low-level kernel: signatures plus a Python reference implementation. From it:

- `dev/generate-kernel-signatures.py` → `include/awkward/kernels.h` and `src/awkward/_cpp/_kernel_signatures.py`.
- `dev/generate-tests.py` + `kernel-test-data.json` → the `tests-spec*` and `tests-cpu-kernels*` suites.

The hand-written C++ implementations live in `src/cpu-kernels/` (one file per kernel) and must match the spec; `nox -s diagnostics` checks kernel definitions. Adding/changing a kernel means touching the YAML spec, the C++ implementation, and (sometimes) test data, then rebuilding (which regenerates) or running `nox -s prepare`.

**`kernel-specification.yml` is the single source of truth for all backends** — the canonical signature, semantics, edge-case behavior, and Python reference. The CPU C++ kernels must conform to it: Awkward guarantees cross-backend consistency (NumPy, typetracer, compiled kernels), so any divergence breaks slicing, broadcasting, masking, union, and Dask typetracer guarantees. Do **not** reorder arguments, change pointer/buffer types, alter edge cases, add kernels directly in C++, or bypass the generation pipeline.

To change a kernel: (1) edit `kernel-specification.yml`; (2) run the generation pipeline (rebuild, or `nox -s prepare -- --signatures --tests`); (3) update the CPU C++ implementation to match the reference; (4) confirm `tests-spec` and `tests-cpu-kernels` pass.

### header-only C++

`header-only/` contains standalone C++ headers (`LayoutBuilder`, `GrowableBuffer`) used by downstream C++ projects; `dev/copy-cpp-headers.py` (part of `prepare`) copies them into `src/awkward/_connect/header-only`, which the package ships for cppyy and cling; the C++ build uses `header-only/` directly. Never edit the copies.

## Conventions and gotchas

- PR titles follow Conventional Commits (`feat:`, `fix:`, `docs:`, ...) — PRs are squash-merged, so the title becomes the commit message. Branches are named `<github-userid>/description`.
- `main` must always be in a working state. Do not commit directly to `main` (PRs only).
- Changes to core dependencies (NumPy, Numba, Pandas, PyArrow) or Python version support must be discussed with maintainers first.
- If a PR requires C++ changes, rebuild the package from the checkout to test it. CI builds wheels for all platforms with cibuildwheel when the C++ code or the build changes.
- AI-assisted contributions must be disclosed in the PR description (see CONTRIBUTING.md).

## Docs

```bash
nox -s prepare -- --docs --headers
nox -s docs -- --serve    # requires Doxygen; serves at localhost:8000
```
