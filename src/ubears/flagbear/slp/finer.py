#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: finer.py
#   Author: xyy15926
#   Created: 2024-06-24 14:04:14
#   Updated: 2025-02-08 17:13:56
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any, TypeVar, TYPE_CHECKING
from collections.abc import Iterable

import logging
import os
import sys
import re
from pathlib import Path
from functools import lru_cache
from datetime import datetime, date
if sys.version_info.major >= 3 and sys.version_info.minor >= 9:
    from importlib.resources import files
else:
    from importlib_resources import files


# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")

ROOT_FLAG = {".root", ".git", "makefile", ".svn", ".vscode"}
DATA_MODNAME = "ubears.data"


# %%
@lru_cache
def get_root_path() -> Path:
    """Get the absolute path of the root of current working project.

    The root is remarked by the ROOT_FLAG, namely the first parent directory
    bottom-up containing file or directory with name in ROOT_FLAG will be
    treated as root.
    """
    # if __name__ == "__main__":
    #     path = Path(".").absolute()
    # else:
    #     path = Path(__file__).absolute().parent
    path = Path.cwd()

    cur = path
    while cur != path.root:
        for ele in cur.iterdir():
            if ele.name in ROOT_FLAG:
                return cur
        cur = cur.parent
        # Reach the root directory.
        if cur.name == "":
            break

    return path


@lru_cache
def get_tmp_path() -> Path:
    """Get the absolute path of the `tmp` of the currrent working project.
    """
    return get_root_path() / "tmp"


@lru_cache
def get_assets_path() -> Path:
    """Get the absolute path of the `assets` of the current working project.
    """
    return get_root_path() / "assets"


# %%
@lru_cache
def get_data(fname: str = None) -> Path:
    """Get the data resources from the project.
    """
    if fname is None:
        return files(DATA_MODNAME)
    else:
        fname = fname.replace("/", ".")
        return files(f"{DATA_MODNAME}.{fname}")


# %%
def date_order_mark(
    kstr: str,
    alts: Iterable[str] = None,
    rdmark: str | None = "today",
    incr: int = 1,
) -> str:
    """Generate mark string with date and order.

    Params:
    --------------------------
    kstr: Key mark
    alts: Alternative to get the start point of the order.
    rdmark: The date mark for the file with format `%Y%m%d`.
      "today": Today will be used as default.
      None: All proper date string with format `%Y%m%d` will be fine enough,
        and the latest date will be used.
    incr: The increasement for the order mark.
      `1` will be used as default, namely the next proper mark will be
        returned.
      And `0` will return the latest existing mark among the `alts`.

    Return:
    --------------------------
    Next or the largest mark.
    """
    # The whole date and order-id will be concated together as an integer.
    if rdmark == "today":
        today = date.today().isoformat().replace("-", "")
        max_mark = int(today + "0000")
        rdmark = f"({today})"
    elif rdmark is None:
        rdmark = r"(20\d{2}[01]\d[0123]\d)"
        max_mark = int("20000101" + "0000")
    else:
        max_mark = int(rdmark + "0000")
        rdmark = f"({rdmark})"
    idx = r"(\d{4})"
    fnreg = f"{kstr}_{rdmark}_{idx}"

    alts = [] if alts is None else alts
    for alt in alts:
        mret = re.match(fnreg, alt)
        if mret:
            cdate, cid = mret.groups()
            max_mark = max(max_mark, int(cdate + cid))
    max_mark += incr
    fmark = str(max_mark)
    fdate, foid = fmark[:8], fmark[8:]

    return f"{kstr}_{fdate}_{foid}"


# %%
def tmp_file(
    fname: str = "tmp.tmp",
    dmark: str | None = "today",
    incr: int = 1
) -> Path:
    """Generate file absolute path in TMP dir.

    Params:
    ----------------------
    fname: The identical part of the file name.
    dmark: The date mark for the file with format `%Y%m%d`.
      "today": Today will be used as default.
      None: All proper date string with format `%Y%m%d` will be fine enough,
        and the latest date will be used.
    incr: The increasement for the order mark
      `1` will be used as default, namely the next proper filename will be
        returned.
      And `0` will return the latest existing filename.

    Return:
    ----------------------
    Absolute file path with date and order tag `fname_<DATE>_<ORDER>.ext`.
    """
    # Mkdir if necessary.
    tfname = get_tmp_path() / fname
    if not tfname.parent.exists():
        tfname.parent.mkdir()
    elif not tfname.parent.is_dir():
        newfd = tfname.parent.with_name(tfname.parent.name + "_bak")
        newfd.mkdir()
        tfname = newfd / tfname.name

    # Split basename and extname.
    basename, extname = os.path.splitext(tfname.name)
    tbname = date_order_mark(basename,
                             (fd.name for fd in tfname.parent.iterdir()),
                             dmark,
                             incr)
    tfname = tfname.with_name(tbname + extname)

    return tfname
