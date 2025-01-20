#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_pdsl.py
#   Author: xyy15926
#   Created: 2024-11-11 11:50:28
#   Updated: 2024-11-11 11:51:36
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ubears.flagbear.slp import finer, pdsl
    reload(finer)
    reload(pdsl)

import re
from datetime import date
import sqlalchemy as sa
import pandas as pd
from ubears.flagbear.slp.finer import get_tmp_path, tmp_file
from ubears.flagbear.slp.pdsl import tmp_table, jsonfy_df

TMP_DIR = "pytest_tmpdir"
TMP_FNAME = f"{TMP_DIR}/tmpf.tmp"
TMP_DB = "pytest_tmpdb.db"
TMP_TBL = "tmp_tbl"


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
