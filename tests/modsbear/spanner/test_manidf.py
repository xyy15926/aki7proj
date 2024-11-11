#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_manidf.py
#   Author: xyy15926
#   Created: 2024-06-06 11:19:40
#   Updated: 2024-11-11 14:48:23
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from modsbear.spanner import manidf
    reload(manidf)

from modsbear.spanner.manidf import merge_dfs, pivot_tags, sequeeze_named_columns


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
                        "vals": np.arange(N)})
    df3 = pd.DataFrame({"on": np.arange(N, dtype=np.float_) + 0.2,
                        "by": np.random.choice(by6, N),
                        "vals": np.arange(N)})
    df4 = pd.DataFrame({"on": np.arange(N, dtype=np.float_) + 0.3,
                        "by": np.random.choice(by6, N),
                        "vals": np.arange(N)})
    dfs = [df1, df2, df3, df4]

    merged = merge_dfs(dfs, on="on")
    assert merged.empty

    merged = merge_dfs([df1, df1, df1], on="on")
    assert np.all(merged.iloc[:, 2] == merged.iloc[:, 4])

    merged = merge_dfs(dfs, on="on", tolerance=None)
    assert np.all(merged.iloc[:, 2] == merged.iloc[:, 4])

    merged = merge_dfs(dfs, on="on", by="by", tolerance=None)


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


