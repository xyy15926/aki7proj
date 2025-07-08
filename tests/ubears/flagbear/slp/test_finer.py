#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_finer.py
#   Author: xyy15926
#   Created: 2024-10-24 20:18:21
#   Updated: 2025-07-08 19:42:11
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
TMP_FNAME_REGEX = f"{TMP_DIR}/tmpf_E0006.tmp"
_PTN = r"E\d{4}"
TMP_FNAME_REGEX_PTN = rf"{TMP_DIR}/tmpf_{_PTN}.tmp"
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

    # Regex key string.
    kstr_ptn = r"kstr_E\d{4}"
    kstr = r"kstr_E0001"
    assert date_order_mark(kstr_ptn) == f"{kstr_ptn}_{today}_0001"
    assert (date_order_mark(kstr_ptn, [f"{kstr}_{today}_0006"])
            == f"{kstr}_{today}_0007")
    assert (date_order_mark(kstr_ptn, [f"{kstr}_{today}_0006"], "20200101")
            == f"{kstr_ptn}_20200101_0001")
    assert (date_order_mark(kstr_ptn, [f"{kstr}_{today}_0006"], "20200101", 5)
            == f"{kstr_ptn}_20200101_0005")
    assert (date_order_mark(kstr_ptn, [f"{kstr}_{today}_0006"], None, 5)
            == f"{kstr}_{today}_0011")

    # Regex key string may lead to unpredictable result.
    kstr2 = r"kstr_E0002"
    assert (date_order_mark(kstr_ptn,
                            [f"{kstr}_{today}_0006", f"{kstr2}_{today}_0001"],
                            None, 5)
            == f"{kstr2}_{today}_0011")


# %%
@pytest.fixture(scope="function", autouse=False)
def tmpfile(request):
    # Process before test.
    print(request)
    tfname = tmp_file(TMP_FNAME)
    tfrname = tmp_file(TMP_FNAME_REGEX)
    tfname.touch()
    tfrname.touch()

    # Return the result.
    yield tfname, tfrname

    # Process after test.
    tfname.unlink()
    tfrname.unlink()
    tfdir = get_tmp_path() / TMP_DIR
    if not any(tfdir.iterdir()):
        tfdir.rmdir()


# %%
def test_tmp_file(tmpfile):
    tf, tfr = tmpfile
    assert tf.is_file()
    assert tfr.is_file()

    # Raise FileExistsError since `tf1` exists.
    with pytest.raises(FileExistsError):
        tmp_file(tf / "any_file")

    # `tmp_file` will update order mark automatically.
    ordix = re.search(r"_(\d{4})\.", tf.name).groups()[0]
    nbname = tf.name.replace(ordix, f"{int(ordix) + 1:04}")
    assert tmp_file(TMP_FNAME).name == nbname

    # `tmp_file` can return exact filename with fuzzy regex.
    ordix = re.search(r"_(\d{4})\.", tfr.name).groups()[0]
    nbname = tfr.name.replace(ordix, f"{int(ordix) + 1:04}")
    assert tmp_file(TMP_FNAME_REGEX_PTN).name == nbname
