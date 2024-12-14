#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_pandas.py
#   Author: xyy15926
#   Created: 2024-05-06 14:44:03
#   Updated: 2024-12-14 21:41:16
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from tqdm import tqdm, trange
from trace import Trace

import numpy as np
import pandas as pd


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


# %%
# `DataFrame.groupby().apply` features:
# 1. If DF returned by the callable applied shares the same index with the
#   group-DF, the result of groupby won't add and additional level of Index.
@pytest.mark.pkgs
@pytest.mark.pandas
def test_groupby_apply_index():
    def func_reset_index(df):
        df = df.copy()
        df = df.reset_index(drop=False)
        return df

    def func_keep_index(df):
        df = df.copy()
        return df

    def func_circle_index(df):
        df = df.copy()
        df.index = list(df.index)
        return df

    def func_change_value(df):
        df = df.copy()
        df["GPKey1"] = 1
        return df

    NN = 1000
    data = repeat_df(NN).set_index("GPKey3Rand")

    index_added = data.groupby("GPKey3", group_keys=True).apply(func_reset_index)
    assert index_added.index.nlevels == data.index.nlevels + 1

    # No additional index will be added if the index value are not changed
    # in `apply`, while the original Index will be sorted in a hash way.
    kret = data.groupby("GPKey3", group_keys=True).apply(func_keep_index)
    assert kret.index.nlevels == data.index.nlevels

    # Ditto.
    kret = data.groupby("GPKey3", group_keys=True).apply(func_circle_index)
    assert kret.index.nlevels == data.index.nlevels

    # Ditto.
    kret = data.groupby("GPKey3", group_keys=True).apply(func_change_value)
    assert kret.index.nlevels == data.index.nlevels

    # Ditto.
    kret = data.groupby("GPKey3", sort=False, group_keys=True).apply(func_change_value)
    assert kret.index.nlevels == data.index.nlevels


# %%
# 2. BUG alog with the former:
# If DF returned by callable applied keep the index unchanged, the duplicates
# in the index and the group-key will slow down the process of concatenation.
# And the more duplicated the index is, the lower efficiency the concatenator
# after application act with.
# Problem:
# 1. It seems that the bug results from the sortation but can't be controlled
#   by the parameter `sort`.
# Solution:
# 1. Reset index.
# 2. Pre-sort group-key.
@pytest.mark.skipif(pd.__version__ == '1.4.4',
                    reason="Efficiency demo")
@pytest.mark.pkgs
@pytest.mark.pandas
def test_groupby_apply_efficiency():
    def func_reset_index(df):
        df = df.copy()
        df = df.reset_index()
        return df

    def func_keep_index(df):
        df = df.copy()
        index = df.index.to_list()
        df.index = index
        return df

    def func_dup_index(df):
        df = df.copy()
        df.index = [1, 2] * (len(df) // 2)
        return df

    def func_modified_index(df):
        df = df.copy()
        index = df.index.to_list()
        index[-1] = 999999
        df.index = index
        return df

    tqdm.pandas(desc="Iters")

    NN = 10000
    # data = repeat_df(NN).set_index("GPKey4")
    data = repeat_df(NN).set_index("GPKey4Rand")
    # data = repeat_df(NN).set_index("GPKey3Rand")

    # For group by `GPKey4` and `GPKey4Rand`.
    # Time: 10s -> Many groups, no duplicates.
    ret = data.groupby("GPKey1", sort=False).progress_apply(func_keep_index)
    # Time: 10s -> Many groups, no duplicates
    ret = data.groupby("GPKey1", sort=False).progress_apply(func_reset_index)
    # Time: 20s -> Medium groups, many duplicates
    ret = data.groupby("GPKey2", sort=False).progress_apply(func_keep_index)
    # Time: 3s -> Medium groups, different index.
    ret = data.groupby("GPKey2", sort=False).progress_apply(func_reset_index)
    # Time: 20s -> Some groups, many duplicates
    ret = data.groupby("GPKey3", sort=False).progress_apply(func_keep_index)
    # Time: 20s -> Some groups, many duplicates
    ret = data.groupby("GPKey3Rand", sort=False).progress_apply(func_keep_index)
    # Time: <1s -> Some groups, differenct index.
    ret = data.groupby("GPKey3", sort=False).progress_apply(func_dup_index)
    # Time: <1s
    ret = data.groupby("GPKey3", sort=True).progress_apply(func_dup_index)
    # Time: <1s -> Some groups, many duplicates, pre-sort
    sorted_data = data.sort_values("GPKey3")
    ret = sorted_data.groupby("GPKey3", sort=False).progress_apply(func_keep_index)
    # Time: <1s -> Some groups, different index.
    ret = data.groupby("GPKey3", sort=True).progress_apply(func_modified_index)
    # Time: 20s -> Few groups, many duplicates.
    ret = data.groupby("GPKey5", sort=True).progress_apply(func_keep_index)


# %%
@pytest.mark.pkgs
@pytest.mark.pandas
def test_zero_divide():
    # ZeroDivisionError won't be raised for numeric dtype.
    df = pd.DataFrame(np.arange(10).reshape((5, 2)))
    div_ret = df[1] / df[0]

    # ZeroDivisionError won't be raised for numeric dtype.
    df = pd.DataFrame(np.arange(10).reshape((5, 2)).astype(float))
    div_ret = df[1] / df[0]

    # While the it will be raised for object dtype though nothing is done
    # except dtype change.
    df = pd.DataFrame(np.arange(10).reshape((5, 2))).astype(object)
    assert np.all(df.dtypes == "object")
    assert isinstance(df.iloc[0, 0], int)
    with pytest.raises(ZeroDivisionError):
        div_ret = df[1] / df[0]


# %%
@pytest.mark.pkgs
@pytest.mark.pandas
def test_concat_nan_index():
    ser = pd.Series([1, 2, 3])
    sers = [ser, ser]

    df = pd.concat(sers, keys=[np.nan, np.nan])
    # `np.nan` can't be in MultiIndex in `pd.concat`.
    with pytest.raises(ValueError):
        df = pd.concat(sers, keys=[(np.nan, np.nan), (np.nan, np.nan)])
    with pytest.raises(ValueError):
        df = pd.concat(sers, keys=[(1, np.nan), (1, np.nan)])
    with pytest.raises(ValueError):
        df = pd.concat(sers, keys=[(np.nan,), (np.nan,)])
    with pytest.raises(ValueError):
        df = pd.concat(sers, keys=[(np.nan,), (1,)])

    # But it's legal to construct `pd.MultiIndex` with `np.nan`.
    idxs = []
    for ser in sers:
        for idx in ser.index:
            if np.isscalar(idx):
                idxs.append((np.nan, idx))
            else:
                idxs.append((np.nan, *idx))
    mindex = pd.MultiIndex.from_tuples(idxs)
    df = pd.concat(sers)
    df.index = mindex
