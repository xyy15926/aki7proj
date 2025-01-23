#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_pandas.py
#   Author: xyy15926
#   Created: 2024-05-06 14:44:03
#   Updated: 2025-01-22 19:52:40
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from packaging.version import Version

from tqdm import tqdm
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
@pytest.mark.skipif(Version(pd.__version__) >= Version("2.0"),
                    reason="Index level bug has been fixed for Pandas == 2.2.3.")
@pytest.mark.pkgs
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
    ridata = data.reset_index(drop=True)

    index_added = data.groupby("GPKey3", group_keys=True).apply(func_reset_index)
    assert index_added.index.nlevels == data.index.nlevels + 1

    # No additional index will be added if the index value are not changed
    # in `apply` with **one by one comparision**, while the original Index will
    # be sorted in a hash way, even though `group_keys` is set.
    kret = data.groupby("GPKey3", group_keys=True).apply(func_keep_index)
    assert kret.index.nlevels == data.index.nlevels
    nkret = data.groupby("GPKey3", group_keys=False).apply(func_keep_index)
    assert np.all(kret == nkret)

    # Ditto.
    rkret = ridata.groupby("GPKey3", group_keys=True).apply(func_keep_index)
    assert rkret.index.nlevels == ridata.index.nlevels
    kret = data.groupby("GPKey3", group_keys=True).apply(func_circle_index)
    assert kret.index.nlevels == data.index.nlevels

    # Ditto.
    kret = data.groupby("GPKey3", group_keys=True).apply(func_change_value)
    assert kret.index.nlevels == data.index.nlevels

    # Ditto.
    kret = data.groupby("GPKey3", sort=False, group_keys=True).apply(func_change_value)
    assert kret.index.nlevels == data.index.nlevels


# %%
# 2. BUG along with the former:
# If DF returned by callable applied keep the index unchanged, the duplicates
# in the (index and the group-key) will slow down the process of concatenation.
# And the more duplicated the index is, the lower efficiency the concatenator
# after application act with.
# Problem:
# 1. It seems that the bug results from the sortation but can't be controlled
#   by the parameter `sort` and it may be hard to sort duplicates. As the
#   traceback always stop at `libops.scalar_compare` of `Concatenator`.
# Solution:
# 1. Reset index in apply function.
# 2. Pre-sort group-key may also be useful. I found that the apply progress
#   may also be much more time-consuming in larger scale of test cases.
@pytest.mark.skip(reason="Efficiency test cases for Pandas == 1.4.4.")
@pytest.mark.pkgs
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

    def func_ori_index(df):
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
    ridata = repeat_df(NN)
    ui = np.arange(len(ridata))
    np.random.shuffle(ui)
    uidata = ridata.set_axis(ui, axis=0)
    r4data = ridata.set_index("GPKey4Rand")
    r3data = ridata.set_index("GPKey3Rand")
    o5data = ridata.set_index("GPKey5")
    o2data = ridata.set_index("GPKey2")
    # data = repeat_df(NN).set_index("GPKey3Rand")

    # There are more and more duplicates from GPKey1 to GPKey5.
    # 1. Noduplicated group key will always be efficient.
    ret = r4data.groupby("GPKey1", sort=False).progress_apply(func_reset_index)
    ret = r4data.groupby("GPKey1", sort=False).progress_apply(func_keep_index)
    ret = uidata.groupby("GPKey1", sort=False).progress_apply(func_keep_index)
    ret = o5data.groupby("GPKey1", sort=False).progress_apply(func_keep_index)

    # 2. Duplicated group key with unique index may also be efficient?
    ret = uidata.groupby("GPKey3", sort=False).progress_apply(func_keep_index)
    ret = ridata.groupby("GPKey3", sort=False).progress_apply(func_keep_index)
    ret = uidata.groupby("GPKey3", sort=False).progress_apply(func_ori_index)
    ret = ridata.groupby("GPKey3", sort=False).progress_apply(func_ori_index)

    # 3. Duplicated group key with duplicated index but different index of
    # the return from `apply` will also be efficient.
    ret = r4data.groupby("GPKey3", sort=False).progress_apply(func_reset_index)
    ret = r4data.groupby("GPKey3", sort=False).progress_apply(func_dup_index)
    ret = r4data.groupby("GPKey3", sort=False).progress_apply(func_modified_index)

    # 4. Duplicated group key with duplicated index and **the same** index of
    # the return from `apply` will be time-comsuming.
    # 4.1 And the overhead will increase along with the duplication-level of
    #   both the index and the group-key.
    ret = r3data.groupby("GPKey5", sort=False).progress_apply(func_keep_index)
    ret = r3data.groupby("GPKey5", sort=True).progress_apply(func_keep_index)
    gpk5 = pd.Index(ret["GPKey5"].values)
    assert not (gpk5.is_monotonic_increasing or gpk5.is_monotonic_decreasing)
    ret = r3data.groupby("GPKey4Rand", sort=False).progress_apply(func_keep_index)
    ret = r4data.groupby("GPKey3", sort=False).progress_apply(func_keep_index)
    ret = r4data.groupby("GPKey3Rand", sort=False).progress_apply(func_keep_index)
    # 4.2 The duplication-level of index seems to be more-weighted.
    # As `o5data.groupby("GPKey2")` can't even return in time when NN > 2000.
    ret = o5data.groupby("GPKey2", sort=False).progress_apply(func_keep_index)
    o5data_s2 = o5data.sort_values("GPKey2")
    ret = o5data_s2.groupby("GPKey2", sort=False).progress_apply(func_keep_index)
    ret = o2data.groupby("GPKey5", sort=False).progress_apply(func_keep_index)


# %%
@pytest.mark.pkgs
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


# %%
@pytest.mark.skipif(Version(pd.__version__) > Version("2.0"),
                    reason="Pandas change the way to handle datetime dtype.")
@pytest.mark.pkgs
def test_empty_datetime_seris_pd14():
    a = pd.Series(["2024-01-01", "2024-01-02"], dtype="M8[s]")
    b = pd.Series(["2023-01-01", "2023-01-02"], dtype="M8[s]")
    gap = a - b
    # `gap` won't inherit the dtype from `a` and `b`.
    assert gap.dtype == "m8[ns]"

    a = pd.Series(dtype="M8[s]")
    b = pd.Series(dtype="M8[s]")
    gap = a - b
    assert gap.dtype == "m8[ns]"

    a = pd.Series(["2024-01-01", "2024-01-02"], dtype="M8[s]").dt.to_period("M")
    b = pd.Series(["2023-01-01", "2023-01-02"], dtype="M8[s]").dt.to_period("M")
    gap = a - b
    assert gap.dtype == "O"

    a = pd.Series(dtype="M8[s]").dt.to_period("M")
    gap = a - b
    assert gap.dtype == "M8[ns]"

    b = pd.Series(dtype="M8[s]").dt.to_period("M")
    # The `gap.dtype` will be float if both are empty for Pandas == 1.4.4.
    gap = a - b
    assert gap.dtype == "float"


# %%
@pytest.mark.skipif(Version(pd.__version__) < Version("2.0"),
                    reason="Pandas change the way to handle datetime dtype.")
@pytest.mark.pkgs
def test_empty_datetime_seris_pd20():
    a = pd.Series(["2024-01-01", "2024-01-02"], dtype="M8[s]")
    b = pd.Series(["2023-01-01", "2023-01-02"], dtype="M8[s]")
    gap = a - b
    assert gap.dtype == "m8[s]"

    a = pd.Series(dtype="M8[s]")
    b = pd.Series(dtype="M8[s]")
    gap = a - b
    assert gap.dtype == "m8[s]"

    a = pd.Series(["2024-01-01", "2024-01-02"], dtype="M8[s]").dt.to_period("M")
    b = pd.Series(["2023-01-01", "2023-01-02"], dtype="M8[s]").dt.to_period("M")
    gap = a - b
    assert gap.dtype == "O"

    a = pd.Series(dtype="M8[s]").dt.to_period("M")
    gap = a - b
    assert gap.dtype == "O"

    b = pd.Series(dtype="M8[s]").dt.to_period("M")
    # And the additional dtype check leads to TypeError for Pandas == 2.2.3.
    with pytest.raises(TypeError):
        gap = a - b


# %%
@pytest.mark.pkgs
def test_na_argmax():
    a = pd.Series([np.nan] * 3, index=[5, 6, 7])
    with pytest.warns(RuntimeWarning, match="All-NaN"):
        assert np.isnan(np.nanmax(a))

    if Version(pd.__version__) > Version("2.0"):
        with pytest.warns(FutureWarning, match="argmax/argmin"):
            assert a.argmax() == -1
        # `np.argmax` will call `pd.Series.argmax`.
        with pytest.warns(FutureWarning, match="argmax/argmin"):
            assert np.argmax(a) == -1
    else:
        assert a.argmax() == -1

    # Pandas will skip NA automatically.
    a = pd.Series([1, np.nan, np.nan], index=[5, 6, 7])
    assert a.argmax() == 0
    
    
