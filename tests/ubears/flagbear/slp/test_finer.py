#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_finer.py
#   Author: xyy15926
#   Created: 2024-10-24 20:18:21
#   Updated: 2025-02-25 18:42:21
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ubears.flagbear.slp import finer
    reload(finer)

import re
from datetime import date
import sqlalchemy as sa
import pandas as pd
from ubears.flagbear.slp.finer import date_order_mark, tmp_file, get_tmp_path
from ubears.flagbear.slp.pdsl import tmp_table, jsonfy_df

TMP_DIR = "pytest_tmpdir"
TMP_FNAME = f"{TMP_DIR}/tmpf.tmp"
TMP_DB = "pytest_tmpdb.db"
TMP_TBL = "tmp_tbl"


# %%
def test_date_order_mark():
    today = date.today().isoformat().replace("-", "")
    kstr = "kstr"
    assert date_order_mark(kstr) == f"{kstr}_{today}_0001"
    assert (date_order_mark(kstr, [f"{kstr}_{today}_0006"])
            == f"{kstr}_{today}_0007")
    assert (date_order_mark(kstr, [f"{kstr}_{today}_0006"], "20200101")
            == f"{kstr}_20200101_0001")
    assert (date_order_mark(kstr, [f"{kstr}_{today}_0006"], "20200101", 5)
            == f"{kstr}_20200101_0005")
    assert (date_order_mark(kstr, [f"{kstr}_{today}_0006"], None, 5)
            == f"{kstr}_{today}_0011")


# %%
@pytest.fixture(scope="function", autouse=False)
def tmpfile(request):
    print(request)
    tfname = tmp_file(TMP_FNAME)
    tfname.touch()
    yield tfname
    tfname.unlink()
    tfdir = get_tmp_path() / TMP_DIR
    if not any(tfdir.iterdir()):
        tfdir.rmdir()


# %%
def test_tmp_file(tmpfile):
    assert tmpfile.is_file()
    # Raise error if mkdir failed.
    with pytest.raises(FileExistsError):
        tmp_file(tmpfile / "any_file")
    ordix = re.search(r"_(\d{4})\.", tmpfile.name).groups()[0]
    nbname = tmpfile.name.replace(ordix, f"{int(ordix) + 1:04}")
    assert tmp_file(TMP_FNAME).name == nbname
