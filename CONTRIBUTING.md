# Contributing to Awkward Array Lite

Thank you for your interest in contributing! We're eager to see your ideas and look forward to working with you.

This document describes the technical procedures we follow in this project. We are all obliged to maintain a welcoming, harassment-free environment.

### Where to start

The front page of this project is its [README](README.md). It includes instructions for [compiling for development](README.md#installation-for-developers).

### Reporting issues

The first thing you should do if you want to fix something is to [submit an issue through GitHub](https://github.com/CoffeaTeam/awkward-lite/issues). That way, we can all see it and maybe one of us or a member of the community knows of a solution that could save you the time spent fixing it. If you "assign yourself" to the issue (top of right side-bar), you can signal your intent to fix it in the issue report.

### Contributing a pull request

Feel free to [open pull requests in GitHub](https://github.com/CoffeaTeam/awkward-lite/pulls) from your [forked repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo) when you start working on the problem. We recommend opening the pull request early so that we can see your progress and communicate about it. (Note that you can `git commit --allow-empty` to make an empty commit and start a pull request before you even have new code.)

Please [make the pull request a draft](https://github.blog/2019-02-14-introducing-draft-pull-requests/) to indicate that it is in an incomplete state and shouldn't be merged until you click "ready for review."

### AI-assisted contributions

We welcome the use of AI tools as part of the development process. They can be valuable aids for drafting, refactoring, documentation, and exploration. However, contributions to Awkward Array require human judgment, contextual understanding, and familiarity with the project’s structure, goals, and standards.

When using AI tools:

* **You remain the author of the contribution.** Review, understand, and test all AI-assisted code or documentation before submitting it under your name. You should be able to explain and defend the changes on request.
* **Avoid fully automated submissions.** Issues or pull requests generated end-to-end by automated tools, without meaningful human review or intent, are not appropriate.
* **Be respectful of reviewers’ time.** Ensure that both the content of the PR and its description reflect your own understanding. Reviewers should not be expected to infer authorship or unknowingly interact with an AI during review.
* **Disclose significant AI assistance.** If AI tools were used for a substantial portion of the contribution, please note this in the PR description. (This guide is an example of that: we used ChatGPT to help with the writing, and in this comment, we acknowledge that fact.)

Contributors are responsible for the correctness, maintainability, and long-term impact of all submitted changes, regardless of whether AI tools were used.

Pull requests that do not meet these expectations may be closed without review.

### Getting your pull request reviewed

Comment in GitHub that you want a review and we'll see it. Only one review is required to be allowed to merge a pull request. We'll work with you to get it into shape.

If you're waiting for a response and haven't heard in a few days, it's possible that we forgot/got distracted/thought someone else was reviewing it/thought we were waiting on you, rather than you waiting on us—just write another comment to remind us.

### Git practices

Unless you ask us not to, we might commit directly to your pull request as a way of communicating what needs to be changed. That said, most of the commits on a pull request are from a single author: corrections and suggestions are exceptions.

Therefore, we prefer git branches to be named with your GitHub userid, such as `ianna/write-contributing-md`.

The titles of pull requests (and therefore the merge commit messages) should follow [these conventions](https://www.conventionalcommits.org/en/v1.0.0/#summary). Mostly, this means prefixing the title with one of these words and a colon:

  * feat: new feature
  * fix: bug-fix
  * perf: code change that improves performance
  * refactor: code change that neither fixes a bug nor adds a feature
  * style: changes that do not affect the meaning of the code
  * test: adding missing tests or correcting existing tests
  * build: changes that affect the build system or external dependencies
  * docs: documentation only changes
  * ci: changes to our CI configuration files and scripts
  * chore: other changes that don't modify src or test files
  * revert: reverts a previous commit

Almost all pull requests are merged with the "squash and merge" feature, so details about commit history within a pull request are hidden from the `main` branch's history. Feel free, therefore, to commit with any frequency you're comfortable with.

It is unnecessary to manually edit (rebase) commit history within a pull request.

### Building and testing locally

The [installation for developers](README.md#installation-for-developers) procedure is described in brief on the front page, and in more detail here.

Awkward Array is shipped as two packages: `awkward` and `awkward-cpp`. The `awkward-cpp` package contains the compiled C++ components required for performance, and `awkward` is only Python code. If you do not need to modify any C++ (the usual case), then `awkward-cpp` can simply be installed using `pip` or `conda`.

Building `awkward-cpp` requires generated code and datafiles (kernel specification, header-only includes, kernel tests). These are generated automatically when `awkward-cpp` is built from the repository — CMake runs the generation scripts at configure time — so no separate preparation step is needed to build the packages.

<details>

The generated files can also be created without building anything, using the `prepare` [nox](https://nox.thea.codes/) session:

```bash
nox -s prepare
```

This is useful e.g. for generating the kernel documentation or the kernel test suites without compiling the C++ package. The `prepare` session accepts flags to specify exact generation targets:
```bash
nox -s prepare -- --tests --docs
```

`nox` also lets us reuse the virtualenvs that it creates for each session with the `-R` flag, eliminating the dependency reinstall time:
```bash
nox -R -s prepare
```

</details>

#### Installing the `awkward-cpp` package

The C++ components can be installed by building the `awkward-cpp` package:

```bash
python -m pip install ./awkward-cpp
```

<details>

If you are working on the C++ components of Awkward Array, it might be more convenient to skip the build isolation step, which involves creating an isolated build environment. First, you must install the build requirements:

```bash
python -m pip install "scikit-build-core>=1.0" pybind11 ninja cmake pyyaml numpy
```

Then the installation can be performed without build isolation:

```bash
python -m pip install --no-build-isolation --check-build-dependencies ./awkward-cpp
```

 </details>

#### Installing the `awkward` package

With `awkward-cpp` installed, an editable installation of the pure-python `awkward` package can be performed with

```bash
python -m pip install -e .
```

#### Testing the installed packages

Finally, let's run the integration test suite to ensure that everything's working as expected:

```bash
python -m pytest -n auto tests
```

For more fine-grained testing, we also have tests of the low-level kernels, which can be invoked with

```bash
python -m pytest -n auto awkward-cpp/tests-spec
python -m pytest -n auto awkward-cpp/tests-cpu-kernels
```

These test suites are generated by building `awkward-cpp` from the repository (or by the `nox -s prepare` session's `--tests` target).

#### Unit tests for the kernels

You can also run additional unit tests that have more test coverage for all the low-level kernels for even more detailed fine-grained testing.

For Python Kernels:
```bash
python -m pytest -n auto awkward-cpp/tests-spec-explicit
```

For CPU Kernels:
```bash
python -m pytest -n auto awkward-cpp/tests-cpu-kernels-explicit
```


### Building wheels

Sometimes it's convenient to build a wheel for the `awkward-cpp` package, so that subsequent re-installs do not require the package to be rebuilt. The `build` package can be used to do this, though care must be taken to specify the *current* Python interpreter in [pipx](https://pypa.github.io/pipx/):

```bash
pipx run --python=$(which python) build --wheel awkward-cpp
```

The built wheel will then be available in `awkward-cpp/dist`.

### Automatic formatting and linting

The Awkward Array project uses [pre-commit](https://pre-commit.com) to handle formatters and linters. This automatically checks (and may push corrections to) your pull request's git branch.

To respond more quickly to pre-commit's feedback, it can help to install it and run it locally. Once it is installed, run

```bash
pre-commit run -a
```

to test all of your files. If you leave off the `-a`, it will run only on currently stashed changes.

### Automated tests

As stated above, we use [pytest](https://docs.pytest.org/) to verify the correctness of the code, and GitHub will reject a pull request if either pre-commit or pytest fails (red "X"). All tests must pass for a pull request to be accepted.

Note that if a pull request doesn't modify code, only the documentation tests will run. That's okay: documentation-only pull requests only need the documentation tests to pass.

### Testing practices

Unless you're refactoring code, such that your changes are fully tested by the existing test suite, new code should be accompanied by new tests. Our testing suite is organized by GitHub issue or pull request number: that is, test file names are

```
tests/test_XXXX-yyyy.py
```

where `XXXX` is either the number of the issue your pull request fixes or the number of the pull request and `yyyy` is descriptive text, often the same as the git branch. This makes it easier to run your test in isolation:

```bash
python -m pytest tests/test_XXXX-yyyy.py
```

and it makes it easier to figure out why a particular test was added. The easiest way to make a new testing file is to copy an existing one and replace its `test_zzzz` functions with your own. The previous tests should also give you a sense of the way we test things and the kinds of things that are constrained in tests.

### Building documentation locally

This section describes how to build the documentation locally.

We use [Sphinx](https://pypi.org/project/Sphinx/) to generate documentation. You may need to install some additional packages:

  * [Doxygen](https://www.doxygen.nl/download.html)
  * [pycparser](https://pypi.org/project/pycparser/)
  * [black](https://pypi.org/project/black/)
  * [sphinx](https://pypi.org/project/sphinx/)
  * [sphinx-rtd-theme](https://pypi.org/project/sphinx-rtd-theme/)

To build documentation locally, first prepare the generated data files with

```bash
nox -s prepare
```

<details>

Only the `--headers` and `--docs` flags are actually required at the time of writing. These can be passed with:

```bash
nox -s prepare -- --docs --headers
```

 </details>

Then, use `nox` to run the various documentation build steps

```bash
nox -s docs
```

This command executes multiple custom Python scripts (some require a working internet connection), in addition to using Sphinx and Doxygen to generate the required browser viewable documentation.

To view the built documentation, open

```bash
docs/_build/html/index.html
```

from the root directory of the project in your preferred web browser, e.g.

```bash
python -m http.server 8080 --directory docs/_build/html/
```

Before re-building documentation, you might want to delete the files that were generated to create viewable documentation. A simple command to remove all of them is

```bash
rm -rf docs/reference/generated docs/_build docs/_static/doxygen
```

There is also a cache in the `docs/_build/.jupyter_cache` directory for Jupyter Book, which can be removed.

### The main branch

The Awkward Array `main` branch must be kept in an unbroken state. There are two reasons for this: so that developers can work independently on known-to-be-working states and so that users can test the latest changes (usually to see if the bug they've discovered is fixed by a potential correction).

Committing directly to `main` is not allowed except for

   * updating the `pyproject.toml` file to increase the version number, which should be independent of pull requests
   * updating documentation or non-code files
   * unprecedented emergencies

and only by the the [reviewing team](CONTRIBUTING.md#getting-your-pull-request-reviewed).

### Guidelines for Dependency and Version Changes

To ensure stability and reproducibility across the Awkward ecosystem:

1. Collaborative Review Required
   - Pull requests that affect core dependencies (e.g., Numba, Pandas, PyArrow, NumPy) or Python version support must be discussed with maintainers before merging.
   - This avoids unilateral decisions that could break critical workflows.

2. Compatibility Transparency
   - If a dependency is not yet available for a new Python release, note this clearly in the PR description and documentation.
   - Example: “Supports Python 3.14, but Numba-dependent features are disabled until upstream support is released.”

3. Testing Expectations
   - CI should run against all supported Python versions.
   - Tests that depend on unsupported features (e.g., Numba on Python 3.14) must be marked with xfail so contributors see the limitation.

4. Version Pinning
   - Pin or constrain versions of critical dependencies to prevent silent breakages from upstream changes.
   - Floating versions may be allowed for secondary utilities, but core scientific packages should be locked.

5. Communication
   - Use GitHub Discussions or Issues to propose changes before opening a PR that alters compatibility.
   - This ensures consensus and avoids surprises for downstream users.



---

#### Example PR Checklist

- Have I discussed this change with maintainers?

- Did I document limitations (e.g., missing Numba support)?

- Are CI tests updated to reflect compatibility status?

- Did I pin or constrain critical dependencies?
