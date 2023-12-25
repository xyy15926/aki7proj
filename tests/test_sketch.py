#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_sketch.py
#   Author: xyy15926
#   Created: 2023-12-21 20:36:50
#   Updated: 2023-12-22 08:47:04
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ragsbear import sketch
    reload(sketch)

import numpy as np
import pandas as pd
from datetime import date

from ragsbear.sketch import DataSketch


# %%
def mock_data(row_n=100):
    seed = 777
    np.random.seed(seed)
    nan_ratio = 0.01

    range_int = np.arange(1, row_n + 1)
    range_int_half = np.concatenate([np.arange(1, row_n // 2 + 1),
                                     np.arange(1, row_n // 2 + 1)])

    int_01 = np.random.randint(0, 1, row_n)
    int_02 = np.random.randint(0, 2, row_n)
    int_03 = np.random.randint(0, 3, row_n)
    int_05 = np.random.randint(0, 5, row_n)
    int_10 = np.random.randint(0, 10, row_n)

    int_str = np.random.choice(list("123456789"), row_n)
    mixed_int_str = np.random.choice(list("123456789ab"), row_n)

    float_10 = np.random.randint(1, 10, row_n)
    float_10 = float_10 + np.random.random(row_n).round(1)

    nan_float_10 = np.random.randint(1, 10, row_n).astype(np.float_)
    nan_float_10[np.random.randint(0, row_n, int(nan_ratio * row_n))] = np.nan

    nan_all = np.array([np.nan] * row_n)
    nan_most = np.concatenate([[np.nan] * (row_n - 1), [1]])
    flat_all = np.array([1] * row_n)
    flat_most = np.concatenate([[1] * (row_n - 1), [2]])

    label = np.random.randint(0, 2, row_n)

    return pd.DataFrame.from_dict({
        "range_int": range_int,
        "range_int_half": range_int_half,
        "int_01": int_01,
        "int_02": int_02,
        "int_03": int_03,
        "int_05": int_05,
        "int_10": int_10,
        "int_str": int_str,
        "mixed_int_str": mixed_int_str,
        "float_10": float_10,
        "nan_float_10": nan_float_10,
        "nan_all": nan_all,
        "nan_most": nan_most,
        "flat_all": flat_all,
        "flat_most": flat_most,
    }), pd.Series(label)


# %%
def test_data_sketch():
    df, y = mock_data()
    # sort_keys = {"range_int_half": 1}
    # factors = 10
    # uni_keys = ["range_int_half", ]
    # keep = keep_keys = "first"
    # na_thresh = 0.9
    # flat_thresh = 0.9
    ds = DataSketch(df, y)
    rint = np.random.randint(1, 100)
    log_dest = f"sketch_{date.today()}_{rint}.xlsx"
    ds.auto_data(log_dest)
