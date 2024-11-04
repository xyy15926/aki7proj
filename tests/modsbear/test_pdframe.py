#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_pdframe.py
#   Author: xyy15926
#   Created: 2024-08-24 18:37:53
#   Updated: 2024-08-25 16:52:12
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from modsbear import pdframe
    from suitbear import finer
    reload(pdframe)
    reload(finer)

from modsbear.pdframe import extract_tables, format_table
from suitbear.finer import get_assets_path


# %%
def test_format_table():
    file = get_assets_path() / "cashflow/wechat_cashflow_demo.pdf"
    table = extract_tables(file)[0]

    columns = {
        "交易单号": None,
        "交易时间": None,
        "交易类型": None,
        "收/支/其他": None,
        "交易方式": None,
        "金额(元)": None,
        "交易对方": None,
        "商户单号": None
    }
    columns = list(columns.keys())
    dtypes = {
        "金额(元)": ("FLOAT", True, 0),
        "交易时间": "DATE",
    }

    ftable, desc = format_table(table, columns,
                                drop_chars="\n\t",
                                dtypes=dtypes)
    assert np.all(ftable.columns == columns)
    assert ftable["金额(元)"].dtype == "float64"
    assert ftable["交易时间"].dtype == "datetime64[ns]"


