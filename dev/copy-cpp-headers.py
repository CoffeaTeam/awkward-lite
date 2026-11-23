# BSD 3-Clause License; see https://github.com/CoffeaTeam/awkward-lite/blob/main/LICENSE

"""Copy the header-only cpp headers into the awkward package, which ships them for cppyy and cling"""

import pathlib
import shutil

root_path = pathlib.Path(__file__).absolute().parents[1]


if __name__ == "__main__":
    header_only_path = root_path / "header-only"

    # Copy to Python package under `awkward`
    connect_path = (
        root_path / "src" / "awkward" / "_connect" / "header-only" / "awkward"
    )
    if connect_path.exists():
        shutil.rmtree(connect_path)
    connect_path.mkdir(parents=True)
    # copy2 preserves source mtimes, so re-running doesn't trigger rebuilds
    for path in header_only_path.rglob("*/awkward/*.h"):
        dest_path = connect_path / path.name
        shutil.copy2(path, dest_path)
