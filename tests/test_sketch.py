#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_sketch.py
#   Author: xyy15926
#   Created: 2023-12-21 20:36:50
#   Updated: 2024-01-18 09:05:34
#   Description:
# ---------------------------------------------------------

# %%
from pytest import mark

if __name__ == "__main__":
    from importlib import reload
    from ragsbear import sketch
    from ringbear import metrics
    from ringbear import sortable
    reload(sketch)
    reload(metrics)
    reload(sortable)

import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import contingency

from ragsbear.sketch import DataSketch
from ringbear.metrics import cal_woes
from ringbear.sortable import tree_cut


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
    # na_thresh = 0.9
    # flat_thresh = 0.9
    ds = DataSketch(df, y, factors, sort_keys, uni_keys, uni_keep)
    ds.check_index()
    colname = "nan_float_10"

    ds.drop_duplicates()
    assert ds.data.shape[0] == rown // 2
    dnf = ds.data[colname]

    ds.drop_flat()
    assert ds.sketch_logs[ds.stage]["manlog"] is not None

    ds.numlog()
    assert ds.sketch_logs[ds.stage]["numlog"] is not None

    ds.numlog_pcorr()
    assert ds.sketch_logs[ds.stage]["numlog_pcorr"] is not None

    ds.srtlog()
    assert ds.sketch_logs[ds.stage]["srtlog"] is not None

    ds.ordinize(new_stage="ord")
    assert ds.stage == "ord"

    ds.woeze(new_stage="woe")
    assert ds.stage == "woe"
    (re, ce), ctab = contingency.crosstab(dnf, ds.label)
    woes = cal_woes(dnf.values, ds.label.values)[1] * np.log(2)
    collog = ds.sketch_logs["ord"]["manlog"][colname]
    assert np.all(np.isclose(ctab, collog["ctab"]))
    assert np.all(np.isclose(woes, collog["woes"]))


@mark.filterwarnings("ignore: divide by zero encountered")
def test_data_sketch_2():
    rown = 100
    df, y = mock_data(rown)
    df["nan_float_10_copy"] = df["nan_float_10"]
    sort_keys = {"range_int_half": 1}
    factors = 10
    uni_keys = ["range_int_half", ]
    uni_keep = "first"
    # na_thresh = 0.9
    # flat_thresh = 0.9
    ds = DataSketch(df, y, factors, sort_keys, uni_keys, uni_keep)
    ds.check_index()
    colname = "nan_float_10"

    ds.drop_duplicates()
    assert ds.data.shape[0] == rown // 2
    dnf = ds.data[colname]

    ds.drop_flat()
    assert ds.sketch_logs[ds.stage]["manlog"] is not None

    ds.numlog()
    assert ds.sketch_logs[ds.stage]["numlog"] is not None

    ds.numlog_pcorr()
    assert ds.sketch_logs[ds.stage]["numlog_pcorr"] is not None

    ds.srtlog()
    assert ds.sketch_logs[ds.stage]["srtlog"] is not None

    ds.ordinize(new_stage="ord")
    assert ds.stage == "ord"

    ds.binize(new_stage="bin")
    assert ds.stage == "bin"
    collog = ds.sketch_logs["ord"]["manlog"][colname]
    edges, ctab = tree_cut(dnf.values, ds.label.values, factors)
    assert np.all(np.isclose(edges, collog["edges"]))
    assert np.all(np.isclose(ctab, collog["ctab"]))

    ds.woeze(new_stage="woe")
    assert ds.stage == "woe"
    ds.drop_flat()
    ds.drop_pcorr()
    assert ds.sketch_logs["woe"]["manlog"] is not None

    # ds.log2excel("ka.xlsx")
