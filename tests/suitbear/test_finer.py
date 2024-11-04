#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_finer.py
#   Author: xyy15926
#   Created: 2024-10-24 20:18:21
#   Updated: 2024-10-29 21:45:03
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import pytest
if __name__ == "__main__":
    from importlib import reload
    from suitbear import finer
    reload(finer)

import re
from datetime import date
import sqlalchemy as sa
import pandas as pd
from suitbear.finer import (get_tmp_path,
                            date_order_mark,
                            tmp_file,
                            tmp_table,
                            jsonfy_df)

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
    ordix = re.search(r"_(\d{4})\.", tmpfile.name).groups()[0]
    nbname = tmpfile.name.replace(ordix, f"{int(ordix) + 1:04}")
    assert tmp_file(TMP_FNAME).name == nbname


# %%
@pytest.fixture(scope="function", autouse=False)
def tmptable(request):
    print(request)
    dbname = tmp_file(TMP_DB)
    fdb = sa.create_engine(f"sqlite:///{dbname}")
    ftbl = tmp_table(fdb, TMP_TBL)
    fdb.execute(f"create table {ftbl} (id int);")
    yield dbname, fdb, ftbl
    dbname.unlink()


# %%
def test_tmp_table(tmptable):
    dbname, fdb, ftbl = tmptable
    assert dbname.is_file()
    ordix = re.search(r"_(\d{4})$", ftbl).groups()[0]
    ntbl = tmp_table(fdb, TMP_TBL)
    assert ftbl.replace(ordix, f"{int(ordix) + 1:04}") == ntbl


# %%
def test_jsonfy_df():
    df = pd.DataFrame([[1, 2], [3, 4]])
    df[1] = [[1, 2], [3, 4]]
    assert pd.api.types.infer_dtype(df[1]) == "mixed"
    dfj = jsonfy_df(df)
    assert pd.api.types.infer_dtype(dfj[1]) == "string"
