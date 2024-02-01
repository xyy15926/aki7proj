#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_dflog.py
#   Author: xyy15926
#   Created: 2024-01-18 20:28:51
#   Updated: 2024-01-26 16:13:44
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark

if __name__ == "__main__":
    from importlib import reload
    from suitbear import dflog
    from ringbear import metrics
    from ringbear import sortable
    reload(dflog)
    reload(metrics)
    reload(sortable)

import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import contingency

from suitbear.dflog import DataProc, serlog, serdiffm
from suitbear.dflog import (RC_NAME, MA_NAME, VA_NAME,
                            PCORR_NAME, ROWN_NAME,
                            NUM_NAME, CAT_NAME,
                            INIT_STATE)
from ringbear.metrics import cal_woes
from ringbear.sortable import tree_cut


# %%
def test_serlog():
    ser = pd.Series([99, 2, 3, "a", 99, None, float('nan')])
    label = pd.Series(np.ones(len(ser), dtype=np.int_))
    uni_df, col_D = serlog(ser, label)
    unis = pd.unique(ser)
    assert len(uni_df) == len(unis) - 1


def test_serdiffm():
    ser = pd.Series([99, 2, 3, "a", 99, None, float('nan')])
    codes = pd.Series(pd.factorize(ser)[0])
    mapper = serdiffm(ser, codes)
    assert np.all(ser.map(mapper).fillna(-1) == codes)


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
@mark.filterwarnings("ignore: divide by zero encountered")
def test_data_sketch():
    rown = 100
    df, y = mock_data(rown)
    sort_keys = {"range_int_half": 1}
    factors = 10
    uni_keys = ["range_int_half", ]
    uni_keep = "first"
    ds = DataProc(df, y, factors, sort_keys, uni_keys, uni_keep)
    logs = ds.logs

    ds.check_index()
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

    ds.ordinize(new_stage="ord")

    ds.woeze(new_stage="woe")
    dfc = ds.data
    vals = logs["woe"][VA_NAME]
    for colname in ds.data:
        dnf = dfc[colname]
        (re, ce), ctab = contingency.crosstab(dnf, ds.label)
        woes, ivs = cal_woes(dnf.values, ds.label.values)
        assert np.all(np.isclose(ctab, vals.loc[colname, [0, 1]]))
        assert np.all(np.isclose(woes, vals.loc[colname, "woes"]))
        assert np.all(np.isclose(ivs, vals.loc[colname, "ivs"]))

    # ds.log2excel("ka.xlsx")


# %%
# Refer to following link for the warning.
# https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
@mark.filterwarnings("ignore: divide by zero encountered")
def test_data_sketch_2():
    rown = 100
    df, y = mock_data(rown)
    sort_keys = {"range_int_half": 1}
    factors = 10
    uni_keys = ["range_int_half", ]
    uni_keep = "first"
    ds = DataProc(df, y, factors, sort_keys, uni_keys, uni_keep)
    logs = ds.logs

    ds.check_index()
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

    ds.ordinize(new_stage="ord")
    ds.binize(new_stage="ord")

    ds.woeze(new_stage="woe")
    dfc = ds.data
    vals = logs["woe"][VA_NAME]
    for colname in ds.data:
        dnf = dfc[colname]
        (re, ce), ctab = contingency.crosstab(dnf, ds.label)
        woes, ivs = cal_woes(dnf.values, ds.label.values)
        assert np.all(np.isclose(ctab, vals.loc[colname, [0, 1]]))
        assert np.all(np.isclose(woes, vals.loc[colname, "woes"]))
        assert np.all(np.isclose(ivs, vals.loc[colname, "ivs"]))
