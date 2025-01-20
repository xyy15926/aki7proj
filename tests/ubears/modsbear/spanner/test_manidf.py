#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_manidf.py
#   Author: xyy15926
#   Created: 2024-06-06 11:19:40
#   Updated: 2025-01-08 20:25:38
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from ubears.modsbear.spanner import manidf
    reload(manidf)

from ubears.flagbear.str2.dups import rename_overlaped
from ubears.modsbear.spanner.manidf import merge_dfs, pivot_tags, sequeeze_named_columns
from ubears.modsbear.spanner.manidf import group_addup_apply


# %%
def repeat_df(NN):
    data = pd.DataFrame()
    data["GPKey1"] = range(NN * 3)
    data["GPKey2"] = list(range(NN)) * 3
    data["GPKey3"] = list(range(NN // 10)) * 3 * 10
    data["GPKey3Rand"] = np.random.randint(NN // 10, size=NN * 3, dtype=int)
    data["GPKey4"] = list(range(NN // 100)) * 3 * 100
    data["GPKey4Rand"] = np.random.randint(NN // 100, size=NN * 3, dtype=int)
    data["GPKey5"] = list(range(NN // 1000)) * 3 * 1000

    return data


def test_group_addup_apply():
    def func_keep_index(df):
        df["addup"] = 1
        return df

    NN = 10000
    # data = repeat_df(NN).set_index("GPKey4")
    ridata = repeat_df(NN)
    r3data = ridata.set_index("GPKey3Rand")
    ret = group_addup_apply(r3data, "GPKey5", func_keep_index)
    assert np.all(ret["addup"] == 1)


# %%
def test_merge_dfs():
    N = 30
    by6 = list("abcdef")
    nby6 = N // len(by6)
    df1 = pd.DataFrame({"on": np.arange(N, dtype=np.float_),
                        "by": sorted(by6 * nby6),
                        "vals": np.arange(N)})
    df2 = pd.DataFrame({"on": np.arange(N, dtype=np.float_) - 0.1,
                        "by": np.random.choice(by6, N),
                        "vals_1": np.arange(N)})
    df3 = pd.DataFrame({"on": np.arange(N, dtype=np.float_) + 0.2,
                        "by": np.random.choice(by6, N),
                        "vals": np.arange(N)})
    df4 = pd.DataFrame({"on": np.arange(N, dtype=np.float_) + 0.3,
                        "by": np.random.choice(by6, N),
                        "vals": np.arange(N)})
    dfs = [df1, df2, df3, df4]

    merged = merge_dfs(dfs, ons="on")
    assert merged.empty

    merged = merge_dfs([df1, df1, df1], ons="on")
    assert np.all(merged.iloc[:, 2] == merged.iloc[:, 4])

    merged = merge_dfs(dfs, ons="on", tolerance=None)
    assert np.all(merged.iloc[:, 2] == merged.iloc[:, 4])

    merged = merge_dfs(dfs, ons="on", bys="by", tolerance=None)


# %%
def test_pivot_tags():
    def count_n(seq: list, sep=","):
        cm = {}
        for ele in seq:
            for i in ele.split(sep):
                cm.setdefault(i, 0)
                cm[i] += 1
        return cm

    ser = pd.Series(["a,b,c", "a,b", "a,c", "c"])
    pt = pivot_tags(ser)
    cn = count_n(ser)
    assert np.all(pt.sum() == pd.Series(cn))

    ser = pd.Series(["a,b,c", "a,b", "a,c", "c,c"])
    pt = pivot_tags(ser)
    cn = count_n(ser)
    assert np.all(pt.sum() == pd.Series(cn))


# %%
def test_sequeeze_named_columns():
    ser = pd.Series(["a,b,c", "a,b", "a,c", "c", None, ""])
    pt = pivot_tags(ser)
    pt["c.1"] = (1 - pt["c"]).astype(int)
    sequeezed = sequeeze_named_columns(pt)
    assert np.all(sequeezed["c"] == 1)


