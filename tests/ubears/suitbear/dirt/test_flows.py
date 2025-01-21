#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_flows.py
#   Author: xyy15926
#   Created: 2024-03-14 10:01:32
#   Updated: 2025-01-20 17:50:17
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark

if __name__ == "__main__":
    from importlib import reload
    from ubears.ringbear import metrics, sortable
    from ubears.modsbear.spanner import dflog
    from ubears.suitbear.dirt import flows
    reload(metrics)
    reload(sortable)
    reload(dflog)
    reload(flows)

import numpy as np
import pandas as pd
from scipy.stats import contingency

from ubears.ringbear.metrics import cal_woes
from ubears.modsbear.spanner.dflog import (
    RC_NAME, MA_NAME, VA_NAME,
    PCORR_NAME, ROWN_NAME,
    NUM_NAME, CAT_NAME,
    INIT_STATE
)
from ubears.suitbear.dirt.flows import DataProc


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

    nan_float_10 = np.random.randint(1, 10, row_n).astype(np.float64)
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
# Refer to the following link for the `FutureWarning` threw in comparion
# between python `str` and `np.numeric`.
# https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
@mark.filterwarnings("ignore: divide by zero encountered")
def test_data_sketch_cols():
    rown = 100
    df, y = mock_data(rown)
    sort_keys = {"range_int_half": 1}
    factors = 10
    uni_keys = ["range_int_half", ]
    uni_keep = "first"
    ds = DataProc(df, y, factors, sort_keys, uni_keys, uni_keep)
    logs = ds.logs

    ds.check_index()
    ds.fillna()
    assert not np.any(pd.isna(ds.data))
    ds.drop_duplicates()
    assert ds.data.shape[0] == rown // 2
    ds.drop_flat()

    dfc = ds.data
    vals = logs[INIT_STATE][VA_NAME]
    for colname in ds.data:
        dnf = dfc[colname]
        (re, ce), ctab = contingency.crosstab(dnf, ds.label)
        woes, ivs = cal_woes(dnf.values, ds.label.values)
        assert np.all(np.isclose(ctab, vals.loc[colname, [0, 1]]))
        assert np.all(np.isclose(woes, vals.loc[colname, "woes"]))
        assert np.all(np.isclose(ivs, vals.loc[colname, "ivs"]))

    return ds


# %%
@mark.filterwarnings("ignore: divide by zero encountered")
def test_data_sketch_drop_pcorr():
    ds = test_data_sketch_cols()
    ds.drop_pcorr()
    assert "int_str" in ds.data.columns


# %%
@mark.filterwarnings("ignore: divide by zero encountered")
def test_data_sketch_woe_directly():
    ds = test_data_sketch_cols()
    logs = ds.logs
    dfc = ds.data

    ds.ordinize(new_stage="ord")
    ds.woeze(new_stage="woe")

    vals = logs["woe"][VA_NAME]
    for colname in ds.data:
        dnf = dfc[colname]
        (re, ce), ctab = contingency.crosstab(dnf, ds.label)
        woes, ivs = cal_woes(dnf.values, ds.label.values)
        assert np.all(np.isclose(ctab, vals.loc[colname, [0, 1]]))
        assert np.all(np.isclose(woes, vals.loc[colname, "woes"]))
        assert np.all(np.isclose(ivs, vals.loc[colname, "ivs"]))


# %%
@mark.filterwarnings("ignore: divide by zero encountered")
def test_data_sketch_woe_after_bin():
    ds = test_data_sketch_cols()
    logs = ds.logs
    dfc = ds.data

    ds.ordinize(new_stage="ord")
    ds.binize(new_stage="ord")
    ds.woeze(new_stage="woe")

    vals = logs["woe"][VA_NAME]
    for colname in ds.data:
        dnf = dfc[colname]
        (re, ce), ctab = contingency.crosstab(dnf, ds.label)
        woes, ivs = cal_woes(dnf.values, ds.label.values)
        assert np.all(np.isclose(ctab, vals.loc[colname, [0, 1]]))
        assert np.all(np.isclose(woes, vals.loc[colname, "woes"]))
        assert np.all(np.isclose(ivs, vals.loc[colname, "ivs"]))
