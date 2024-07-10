#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: finer.py
#   Author: xyy15926
#   Created: 2024-06-24 14:04:14
#   Updated: 2024-06-25 18:10:06
#   Description:
# ---------------------------------------------------------

# %%
from pathlib import Path
from functools import lru_cache

ROOT_FLAG = {".root", ".git", "makefile", ".svn", ".vscode"}


# %%
@lru_cache
def get_root_path() -> Path:
    """Get the absolute path of the root of the project.
    """
    if __name__ == "__main__":
        path = Path(".").absolute()
    else:
        path = Path(__file__).absolute().parent

    cur = path
    while cur != path.root:
        for ele in cur.iterdir():
            if ele.name in ROOT_FLAG:
                return cur
        cur = path.parent
    else:
        return path


@lru_cache
def get_tmp_path() -> Path:
    """Get the absolute path of the tmp of the project.
    """
    return get_root_path() / "tmp"


@lru_cache
def get_assets_path() -> Path:
    """Get the absolute path of the assets of the project.
    """
    return get_root_path() / "assets"
