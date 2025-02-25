#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_callables.py
#   Author: xyy15926arruns
#   Created: 2025-02-20 18:56:36
#   Updated: 2025-02-25 14:08:03
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ubears.flagbear.const import callables
    from ubears.flagbear.llp import parser
    from ubears.modsbear.locale import calender
    reload(callables)
    reload(parser)
    reload(calender)

import numpy as np
import pandas as pd
from ubears.flagbear.const.callables import (
    # Aggregation
    max, min, sum, avg,
    nnfilter, getn,
    drop_duplicates,
    argmax, argmin, argmaxs, argmins,
    flat1_max, coef_var,
    # Transformation
    sortby,
    map, sep_map,
    isin,
    cb_fst, cb_max, cb_min,
    mon_itvl, day_itvl,
    is_busiday, not_busiday, get_hour,
)
from ubears.flagbear.llp.parser import EnvParser


# %% -------------------------------------------------------------------------
#                    * * * Aggregation Callables * * *
# %% -------------------------------------------------------------------------
# %%
def test_nnfilter():
    envp = EnvParser()

    # List, np.ndarray, pd.Series.
    x = [1, 2, 2, None]
    ret = nnfilter(x)
    assert np.all(ret == [1, 2, 2])
    env = dict(x=x)
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert np.all(ret == [1, 2, 2])

    x = np.array([1, 2, 2, None])
    ret = nnfilter(x)
    assert np.all(ret == [1, 2, 2])
    env = dict(x=x)
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert np.all(ret == [1, 2, 2])

    x = pd.Series([1, 2, 2, None])
    ret = nnfilter(x)
    assert np.all(np.isclose(ret, [1, 2, 2, np.nan], equal_nan=True))
    env = pd.DataFrame(dict(x=x))
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert np.all(np.isclose(ret, [1, 2, 2, np.nan], equal_nan=True))

    # Empty sequence.
    ret = nnfilter([])
    assert ret == []
    env = dict(x=[])
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert ret == []


# %%
def test_getn():
    envp = EnvParser()

    # List, np.ndarray, pd.Series.
    x = [1, 2, 3, 4]
    ret = getn(x, 1)
    assert ret == 2
    ret = getn(x, 7)
    assert ret is None
    ret = getn(x, None)
    assert ret is None

    env = dict(x=x)
    ret = envp.bind_env(env).parse("getn(x, 1)")
    assert ret == 2
    ret = envp.bind_env(env).parse("getn(x, 7)")
    assert ret is None
    ret = envp.bind_env(env).parse("getn(x, None)")
    assert ret is None

    x = np.array([1, 2, 3, 4])
    ret = getn(x, 1)
    assert ret == 2
    ret = getn(x, 7)
    assert ret is None
    ret = getn(x, None)
    assert ret is None

    env = dict(x=x)
    ret = envp.bind_env(env).parse("getn(x, 1)")
    assert ret == 2
    ret = envp.bind_env(env).parse("getn(x, 7)")
    assert ret is None
    ret = envp.bind_env(env).parse("getn(x, None)")
    assert ret is None

    x = pd.Series([1, 2, 3, 4])
    ret = getn(x, 1)
    assert ret == 2
    ret = getn(x, 7)
    assert ret is None
    ret = getn(x, None)
    assert ret is None

    env = pd.DataFrame(dict(x=x))
    ret = envp.bind_env(env).parse("getn(x, 1)")
    assert ret == 2
    ret = envp.bind_env(env).parse("getn(x, 7)")
    assert ret is None
    ret = envp.bind_env(env).parse("getn(x, None)")
    assert ret is None

    # Empty sequence.
    ret = getn([], 1)
    assert ret is None
    ret = envp.bind_env(env).parse("getn([], 1)")
    assert ret is None


# %%
def test_dropdup():
    envp = EnvParser()

    # Single List, np.ndarray and pd.Series.
    x = [1, 1, 2, 3, "a", None, None, np.nan, np.nan]
    ret = drop_duplicates([x,])
    # `None` and `np.nan` will be treated as the same, and the first will be
    # reserved.
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)
    ret = drop_duplicates(x)
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)

    env = dict(x=x)
    ret = envp.bind_env(env).parse("drop_duplicates([x,])")
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)
    ret = envp.bind_env(env).parse("drop_duplicates(x)")
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)

    x = np.array([1, 1, 2, 3, "a", None, None, np.nan, np.nan], dtype="O")
    ret = drop_duplicates([x,])
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)
    ret = drop_duplicates(x)
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)

    env = dict(x=x)
    ret = envp.bind_env(env).parse("drop_duplicates([x,])")
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)
    ret = envp.bind_env(env).parse("drop_duplicates(x)")
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)

    x = pd.Series([1, 1, 2, 3, "a", None, None, np.nan, np.nan])
    ret = drop_duplicates([x,])
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)
    ret = drop_duplicates(x)
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)

    env = pd.DataFrame(dict(x=x))
    ret = envp.bind_env(env).parse("drop_duplicates([x,])")
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)
    ret = envp.bind_env(env).parse("drop_duplicates(x)")
    for le,re in zip(ret, [1, 2, 3, "a", None]):
        assert le == re or np.isnan(le)

    # Empty single sequences.
    ret = drop_duplicates([])
    assert len(ret) == 0
    ret = drop_duplicates([[]])
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("drop_duplicates([])")
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("drop_duplicates([[]])")
    assert len(ret) == 0

    # Single factor of multiple sequences.
    ret = drop_duplicates([1, 1])
    assert np.all(ret == [1,])
    assert ret.shape == (1,)
    ret = envp.bind_env(env).parse("drop_duplicates([1, 1])")
    assert np.all(ret == [1,])
    assert ret.shape == (1,)

    # Multiple List, np.ndarray and pd.Series.
    x = [1, 1, 2, 3, "a", None, None, np.nan, np.nan]
    y = [1, 1, 2, "a", 5, 5, 5, 6, 6]
    ret = drop_duplicates([x, y])
    for le,re in zip(ret.ravel(),
                     [1, 1, 2, 2, 3, "a", "a", 5, None, 5, np.nan, 6]):
        assert le == re or np.isnan(le)

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("drop_duplicates([x,y])")
    for le,re in zip(ret.ravel(),
                     [1, 1, 2, 2, 3, "a", "a", 5, None, 5, np.nan, 6]):
        assert le == re or np.isnan(le)

    x = np.array([1, 1, 2, 3, "a", None, None, np.nan, np.nan], dtype="O")
    y = np.array([1, 1, 2, "a", 5, 5, 5, 6, 6], dtype="O")
    ret = drop_duplicates([x, y])
    for le,re in zip(ret.ravel(),
                     [1, 1, 2, 2, 3, "a", "a", 5, None, 5, np.nan, 6]):
        assert le == re or np.isnan(le)

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("drop_duplicates([x,y])")
    for le,re in zip(ret.ravel(),
                     [1, 1, 2, 2, 3, "a", "a", 5, None, 5, np.nan, 6]):
        assert le == re or np.isnan(le)

    x = pd.Series([1, 1, 2, 3, "a", None, None, np.nan, np.nan])
    y = pd.Series([1, 1, 2, "a", 5, 5, 5, 6, 6])
    ret = drop_duplicates([x, y])
    for le,re in zip(ret.ravel(),
                     [1, 1, 2, 2, 3, "a", "a", 5, None, 5, np.nan, 6]):
        assert le == re or np.isnan(le)

    env = pd.DataFrame(dict(x=x, y=y))
    ret = envp.bind_env(env).parse("drop_duplicates([x,y])")
    for le,re in zip(ret.ravel(),
                     [1, 1, 2, 2, 3, "a", "a", 5, None, 5, np.nan, 6]):
        assert le == re or np.isnan(le)

    # Empty multiple sequences.
    ret = drop_duplicates([[], []])
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("drop_duplicates([[], []])")
    assert len(ret) == 0

    # Single factor of multiple sequences.
    ret = drop_duplicates([[1, 1], [1, 1]])
    assert np.all(ret == [[1, 1]])
    assert ret.shape == (1, 2)
    ret = envp.bind_env(env).parse("drop_duplicates([[1, 1], [1, 1]])")
    assert np.all(ret == [[1, 1]])
    assert ret.shape == (1, 2)


# %%
def test_maxminavg():
    envp = EnvParser()

    # List, np.ndarray, pd.Series for max, avg and sum.
    # Skip min.
    x = [1, 2, 2, np.nan]
    ret = max(x)
    assert ret == 2
    ret = avg(x)
    assert ret == 5 / 3
    ret = sum(x)
    assert ret == 5

    env = dict(x=x)
    ret = envp.bind_env(env).parse("max(x)")
    assert ret == 2
    ret = envp.bind_env(env).parse("avg(x)")
    assert ret == 5 / 3
    ret = envp.bind_env(env).parse("sum(x)")
    assert ret == 5

    x = np.array([1, 2, 2, np.nan])
    ret = max(x)
    assert ret == 2
    ret = avg(x)
    assert ret == 5 / 3
    ret = sum(x)
    assert ret == 5

    env = dict(x=x)
    ret = envp.bind_env(env).parse("max(x)")
    assert ret == 2
    ret = envp.bind_env(env).parse("avg(x)")
    assert ret == 5 / 3
    ret = envp.bind_env(env).parse("sum(x)")
    assert ret == 5

    x = pd.Series([1, 2, 2, np.nan])
    ret = max(x)
    assert ret == 2
    ret = avg(x)
    assert ret == 5 / 3
    ret = sum(x)
    assert ret == 5

    env = dict(x=x)
    ret = envp.bind_env(env).parse("max(x)")
    assert ret == 2
    ret = envp.bind_env(env).parse("avg(x)")
    assert ret == 5 / 3
    ret = envp.bind_env(env).parse("sum(x)")
    assert ret == 5

    # Empty sequence for max, avg and sum.
    # Skip min.
    ret = max([])
    assert np.isnan(ret)
    ret = avg([])
    assert np.isnan(ret)
    ret = sum([])
    assert ret == 0

    ret = envp.bind_env(env).parse("max([])")
    assert np.isnan(ret)
    ret = envp.bind_env(env).parse("avg([])")
    assert np.isnan(ret)
    ret = envp.bind_env(env).parse("sum([])")
    assert ret == 0

    # All-nan sequence for max.
    x = [np.nan] * 5
    ret = max(x)
    assert np.isnan(ret)
    ret = avg(x)
    assert np.isnan(ret)
    ret = sum(x)
    assert ret == 0

    env = dict(x=x)
    ret = envp.bind_env(env).parse("max(x)")
    assert np.isnan(ret)
    ret = envp.bind_env(env).parse("avg(x)")
    assert np.isnan(ret)
    ret = envp.bind_env(env).parse("sum(x)")
    assert ret == 0

    # All-nan sequence for max.
    x = [np.inf] * 5
    ret = max(x)
    assert np.isinf(ret)
    ret = avg(x)
    assert np.isinf(ret)
    ret = sum(x)
    assert np.isinf(ret)

    env = dict(x=x)
    ret = envp.bind_env(env).parse("max(x)")
    assert np.isinf(ret)
    ret = envp.bind_env(env).parse("avg(x)")
    assert np.isinf(ret)
    ret = envp.bind_env(env).parse("sum(x)")
    assert np.isinf(ret)


# %%
def test_argmaxmin():
    envp = EnvParser()

    # List, np.ndarray, pd.Series for `argmax`.
    # Skip `argmin`.
    x = [1, 1, 2, 2, np.nan]
    ret = argmax(x)
    assert ret == 2
    env = dict(x=x)
    ret = envp.bind_env(env).parse("argmax(x)")
    assert ret == 2

    x = np.array([1, 1, 2, 2, np.nan])
    ret = argmax(x)
    assert ret == 2
    env = dict(x=x)
    ret = envp.bind_env(env).parse("argmax(x)")
    assert ret == 2

    x = pd.Series([1, 1, 2, 2, np.nan])
    ret = argmax(x)
    assert ret == 2
    env = pd.DataFrame(dict(x=x))
    ret = envp.bind_env(env).parse("argmax(x)")
    assert ret == 2

    # Empty sequence for `argmax`.
    # Skip `argmin`.
    ret = argmax([])
    assert ret is None
    ret = envp.bind_env(env).parse("argmax([])")
    assert ret is None


# %%
def test_argmaxsmins():
    envp = EnvParser()

    # List, np.ndarray, pd.Series of numbers.
    x = [1, 1, 2, 2, np.nan]
    y = [1, 2, 3, 4, 5]
    ret = argmaxs(x, y)
    assert np.all(ret == [3, 4])
    ret = argmins(x, y)
    assert np.all(ret == [1, 2])

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == [3, 4])
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [1, 2])

    x = np.array([1, 1, 2, 2, np.nan])
    y = np.array([1, 2, 3, 4, 5])
    ret = argmaxs(x, y)
    assert np.all(ret == [3, 4])
    ret = argmins(x, y)
    assert np.all(ret == [1, 2])

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == [3, 4])
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [1, 2])

    x = pd.Series([1, 1, 2, 2, np.nan])
    y = pd.Series([1, 2, 3, 4, 5])
    ret = argmaxs(x, y)
    assert np.all(ret == [3, 4])
    ret = argmins(x, y)
    assert np.all(ret == [1, 2])

    env = pd.DataFrame(dict(x=x, y=y))
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == [3, 4])
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [1, 2])

    # All NAs.
    x = np.array([np.nan] * 5)
    y = np.array([1, 2, 3, 4, 5])
    with pytest.warns(RuntimeWarning):
        ret = argmaxs(x, y)
        assert len(ret) == 0

    env = dict(x=x, y=y)
    with pytest.warns(RuntimeWarning):
        ret = envp.bind_env(env).parse("argmaxs(x, y)")
        assert len(ret) == 0

    # np.ndarray, pd.Series of datetime64.
    x = np.array(["2021-12-12",
                  "2021-11-11",
                  "2022-11-11",
                  "2021-11-11",
                  "NaT"], dtype="M8[D]")
    y = [1, 2, 3, 4, 5]
    ret = argmaxs(x, y)
    assert np.all(ret == 3)
    ret = argmins(x, y)
    assert np.all(ret == [2, 4])

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == 3)
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [2, 4])

    x = pd.Series(["2021-12-12",
                   "2021-11-11",
                   "2022-11-11",
                   "2021-11-11",
                   "NaT"], dtype="M8[ms]")
    y = [1, 2, 3, 4, 5]
    ret = argmaxs(x, y)
    assert np.all(ret == 3)
    ret = argmins(x, y)
    assert np.all(ret == [2, 4])

    env = pd.DataFrame(dict(x=x, y=y))
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == 3)
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [2, 4])

    # Empty sequence.
    ret = argmaxs([], [])
    assert len(ret) == 0
    ret = argmins([], [])
    assert len(ret) == 0

    ret = envp.bind_env(env).parse("argmaxs([], [])")
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("argmins([], [])")
    assert len(ret) == 0


# %%
def test_cals():
    envp = EnvParser()

    # `flat1_max` with sequences of difference length.
    x = [1, 1, 1, 1, 0, 4, 5, 0]
    ret = flat1_max(x)
    assert ret == 4
    ret = flat1_max([])
    assert ret == 0
    ret = flat1_max([5])
    assert ret == 1
    ret = flat1_max([0])
    assert ret == 0

    env = dict(x=x)
    ret = envp.bind_env(env).parse("flat1_max(x)")
    assert ret == 4
    ret = envp.bind_env(env).parse("flat1_max([])")
    assert ret == 0
    ret = envp.bind_env(env).parse("flat1_max([5])")
    assert ret == 1
    ret = envp.bind_env(env).parse("flat1_max([0])")
    assert ret == 0

    # Coefficent of variation.
    x = [1, 2, 1, 2, 0]
    ret = coef_var(x)
    assert ret == np.std(x) / np.mean(x)
    ret = coef_var([])
    assert ret == 0

    env = dict(x=x)
    ret = envp.bind_env(env).parse("coef_var(x)")
    assert ret == np.std(env["x"]) / np.mean(env["x"])
    ret = envp.bind_env(env).parse("coef_var([])")
    assert ret == 0


# %% -------------------------------------------------------------------------
#                    * * * Transformation Callables * * *
# %% -------------------------------------------------------------------------
def test_sortby():
    envp = EnvParser()

    # List, np.ndarray and pd.Series.
    x = [11, np.nan, 13, 4, np.nan]
    y = [1, 12, 3, 14, np.nan]
    ret = sortby(x, y)
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("sortby(x, y)")
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    x = np.array([11, np.nan, 13, 4, np.nan])
    y = np.array([1, 12, 3, 14, np.nan])
    ret = sortby(x, y)
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("sortby(x, y)")
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    x = pd.Series([11, np.nan, 13, 4, np.nan])
    y = pd.Series([1, 12, 3, 14, np.nan])
    ret = sortby(x, y)
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    env = pd.DataFrame(dict(x=x, y=y))
    ret = envp.bind_env(env).parse("sortby(x, y)")
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    # Empty sequence.
    ret = sortby([], [])
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("sortby([], [])")
    assert len(ret) == 0


# %%
def test_maps():
    ref = dict(zip("abcde", range(1, 6)))

    def refcall(ele):
        refi = dict(zip("abcde", range(1, 6)))
        return refi.get(ele, 7)

    envp = EnvParser({"ref": ref, "refcall": refcall})

    # Map
    x = ["a", "c", "d", "e", "b", None, "NA"]
    ret = map(x, ref)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True))
    ret = map(x, ref, None)
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = map(x, refcall)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))

    env = dict(x=x)
    ret = envp.bind_env(env).parse("map(x, ref)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("map(x, ref, None)")
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = envp.bind_env(env).parse("map(x, refcall)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))

    x = np.array(["a", "c", "d", "e", "b", None, "NA"])
    ret = map(x, ref)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True))
    ret = map(x, ref, None)
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = map(x, refcall)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))

    x = np.array(["a", "c", "d", "e", "b", None, "NA"])
    env = dict(x=x)
    ret = envp.bind_env(env).parse("map(x, ref)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("map(x, ref, None)")
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = envp.bind_env(env).parse("map(x, refcall)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))

    x = pd.Series(["a", "c", "d", "e", "b", None, "NA"])
    ret = map(x, ref)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True))
    ret = map(x, ref, None)
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = map(x, refcall)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))

    env = pd.DataFrame(dict(x=x))
    ret = envp.bind_env(env).parse("map(x, ref)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("map(x, ref, None)")
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = envp.bind_env(env).parse("map(x, refcall)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, 7, 7], equal_nan=True))

    # Seperate map.
    x = ["a,b,c,d,e,na,g", "a,g,c", "g"]
    ret = sep_map(x, ref)
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = sep_map(x, ref, ",", ":")
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])
    ret = sep_map(x, refcall)
    for le, re in zip(ret, [(1, 2, 3, 4, 5, 7,), (1, 3, 7), (7,)]):
        assert le == re
    ret = sep_map(x, refcall, ",", ":")
    assert np.all(ret == ["1:2:3:4:5:7", "1:3:7", "7"])

    env = dict(x=x)
    ret = envp.bind_env(env).parse("sep_map(x, ref)")
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = envp.bind_env(env).parse('sep_map(x, ref, ",", ":")')
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])
    ret = envp.bind_env(env).parse("sep_map(x, refcall)")
    for le, re in zip(ret, [(1, 2, 3, 4, 5, 7,), (1, 3, 7), (7,)]):
        assert le == re
    ret = envp.bind_env(env).parse('sep_map(x, refcall, ",", ":")')
    assert np.all(ret == ["1:2:3:4:5:7", "1:3:7", "7"])

    x = np.array(["a,b,c,d,e,na,g", "a,g,c", "g"])
    ret = sep_map(x, ref)
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = sep_map(x, ref, ",", ":")
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])
    ret = sep_map(x, refcall)
    for le, re in zip(ret, [(1, 2, 3, 4, 5, 7,), (1, 3, 7), (7,)]):
        assert le == re
    ret = sep_map(x, refcall, ",", ":")
    assert np.all(ret == ["1:2:3:4:5:7", "1:3:7", "7"])

    env = dict(x=x, ref=ref)
    ret = envp.bind_env(env).parse("sep_map(x, ref)")
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = envp.bind_env(env).parse('sep_map(x, ref, ",", ":")')
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])
    ret = envp.bind_env(env).parse("sep_map(x, refcall)")
    for le, re in zip(ret, [(1, 2, 3, 4, 5, 7,), (1, 3, 7), (7,)]):
        assert le == re
    ret = envp.bind_env(env).parse('sep_map(x, refcall, ",", ":")')
    assert np.all(ret == ["1:2:3:4:5:7", "1:3:7", "7"])

    x = pd.Series(["a,b,c,d,e,na,g", "a,g,c", "g"])
    ret = sep_map(x, ref)
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = sep_map(x, ref, ",", ":")
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])
    ret = sep_map(x, refcall)
    for le, re in zip(ret, [(1, 2, 3, 4, 5, 7,), (1, 3, 7), (7,)]):
        assert le == re
    ret = sep_map(x, refcall, ",", ":")
    assert np.all(ret == ["1:2:3:4:5:7", "1:3:7", "7"])

    env = pd.DataFrame(dict(x=x))
    ret = envp.bind_env(env).parse("sep_map(x, ref)")
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = envp.bind_env(env).parse('sep_map(x, ref, ",", ":")')
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])
    ret = envp.bind_env(env).parse("sep_map(x, refcall)")
    for le, re in zip(ret, [(1, 2, 3, 4, 5, 7,), (1, 3, 7), (7,)]):
        assert le == re
    ret = envp.bind_env(env).parse('sep_map(x, refcall, ",", ":")')
    assert np.all(ret == ["1:2:3:4:5:7", "1:3:7", "7"])


# %%
def test_isin():
    envp = EnvParser()

    # List, np.ndarry and pd.Series to check:
    # 1. If the elements in the former sequence are in the latter sequence.
    x = [1, 2, 3, 4, None, np.nan, "a"]
    y = [1, np.nan, "a", None]
    ret = isin(x, y)
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    x = np.array([1, 2, 3, 4, None, np.nan, "a"])
    y = np.array([1, np.nan, "a", None])
    ret = isin(x, y)
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    x = pd.Series([1, 2, 3, 4, None, np.nan, "a"])
    y = pd.Series([1, np.nan, "a", None])
    ret = isin(x, y)
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    env = pd.DataFrame(dict(x=x, y=y))
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 0, 0, 0, 1, 1, 1])

    # 2. If the former scalar is in the elements of the latter sequence.
    x = "a"
    y = ["a", "abc", "bc", None, np.nan, ("a", "b"), ["abc"]]
    ret = isin(x, y)
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])

    x = "a"
    y = np.array(["a", "abc", "bc", None, np.nan, ("a", "b"), ["abc"]],
                 dtype="O")
    ret = isin(x, y)
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])

    x = "a"
    y = pd.Series(["a", "abc", "bc", None, np.nan, ("a", "b"), ["abc"]],
                  dtype="O")
    ret = isin(x, y)
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("isin(x, y)")
    assert np.all(ret == [1, 1, 0, 0, 0, 1, 0])


# %%
def test_cb_fstmaxmin():
    envp = EnvParser()

    # List, np.ndarray, pd.Series of numbers.
    x = [11, np.nan, 13, 4, np.nan]
    y = [np.nan, 12, 3, 14, np.nan]
    ret = cb_fst(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = cb_max(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = cb_min(x, y)
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    x = np.array([11, np.nan, 13, 4, np.nan])
    y = np.array([np.nan, 12, 3, 14, np.nan])
    ret = cb_fst(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = cb_max(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = cb_min(x, y)
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    x = pd.Series([11, np.nan, 13, 4, np.nan])
    y = pd.Series([np.nan, 12, 3, 14, np.nan])
    ret = cb_fst(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = cb_max(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = cb_min(x, y)
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    env = pd.DataFrame(dict(x=x, y=y))
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 3, 4, np.nan], equal_nan=True))

    # np.ndarray, pd.Series of datetime64.
    x = np.array(["2021-11-11",
                  "NaT",
                  "2021-11-13",
                  "2021-11-14",
                  "NaT"], dtype="M8[D]")
    y = np.array(["2021-12-11",
                  "2021-12-12",
                  "2021-12-13",
                  "2021-12-14",
                  "NaT"], dtype="M8[D]")
    ret = cb_fst(x, y)
    assert np.all(ret[:-1] == np.array(["2021-11-11",
                                        "2021-12-12",
                                        "2021-11-13",
                                        "2021-11-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])
    ret = cb_max(x, y)
    assert np.all(ret[:-1] == np.array(["2021-12-11",
                                        "2021-12-12",
                                        "2021-12-13",
                                        "2021-12-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])
    ret = cb_min(x, y)
    assert np.all(ret[:-1] == np.array(["2021-11-11",
                                        "2021-12-12",
                                        "2021-11-13",
                                        "2021-11-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(ret[:-1] == np.array(["2021-11-11",
                                        "2021-12-12",
                                        "2021-11-13",
                                        "2021-11-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(ret[:-1] == np.array(["2021-12-11",
                                        "2021-12-12",
                                        "2021-12-13",
                                        "2021-12-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(ret[:-1] == np.array(["2021-11-11",
                                        "2021-12-12",
                                        "2021-11-13",
                                        "2021-11-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])

    x = pd.Series(["2021-11-11",
                   "NaT",
                   "2021-11-13",
                   "2021-11-14",
                   "NaT"], dtype="M8[ms]")
    y = pd.Series(["2021-12-11",
                   "2021-12-12",
                   "2021-12-13",
                   "2021-12-14",
                   "NaT"], dtype="M8[ms]")
    ret = cb_fst(x, y)
    assert np.all(ret[:-1] == np.array(["2021-11-11",
                                        "2021-12-12",
                                        "2021-11-13",
                                        "2021-11-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])
    ret = cb_max(x, y)
    assert np.all(ret[:-1] == np.array(["2021-12-11",
                                        "2021-12-12",
                                        "2021-12-13",
                                        "2021-12-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])
    ret = cb_min(x, y)
    assert np.all(ret[:-1] == np.array(["2021-11-11",
                                        "2021-12-12",
                                        "2021-11-13",
                                        "2021-11-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])

    env = pd.DataFrame(dict(x=x, y=y))
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(ret[:-1] == np.array(["2021-11-11",
                                        "2021-12-12",
                                        "2021-11-13",
                                        "2021-11-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(ret[:-1] == np.array(["2021-12-11",
                                        "2021-12-12",
                                        "2021-12-13",
                                        "2021-12-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(ret[:-1] == np.array(["2021-11-11",
                                        "2021-12-12",
                                        "2021-11-13",
                                        "2021-11-14"], dtype="M8[D]"))
    assert np.isnat(ret[-1])

    # Object for `cb_first` only.
    x = ["a", None, None, 11, np.nan, 13, 4, np.nan]
    y = [None, "b", None, np.nan, 12, 3, 14, np.nan]
    ret = cb_fst(x, y)
    for le, re in zip(ret, ["a", "b", None,11, 12, 13, 4, np.nan]):
        assert le == re or np.isnan(le)

    x = np.array(["a", None, None, 11, np.nan, 13, 4, np.nan])
    y = np.array([None, "b", None, np.nan, 12, 3, 14, np.nan])
    ret = cb_fst(x, y)
    for le, re in zip(ret, ["a", "b", None,11, 12, 13, 4, np.nan]):
        assert le == re or np.isnan(le)

    x = pd.Series(["a", None, None, 11, np.nan, 13, 4, np.nan])
    y = pd.Series([None, "b", None, np.nan, 12, 3, 14, np.nan])
    ret = cb_fst(x, y)
    for le, re in zip(ret, ["a", "b", None,11, 12, 13, 4, np.nan]):
        assert le == re or np.isnan(le)


# %%
def test_mon_day_itvl():
    envp = EnvParser()

    # List or string, np.ndarray and pd.Series.
    x = ["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"]
    y = ["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"]
    ret = mon_itvl(x, y)
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True))
    ret = day_itvl(x, y)
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True))

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("mon_itvl(x, y)")
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("day_itvl(x, y)")
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True))

    x = np.array(["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"],
                 dtype="M8[D]")
    y = np.array(["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"],
                 dtype="M8[D]")
    ret = mon_itvl(x, y)
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True))
    ret = day_itvl(x, y)
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True))

    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("mon_itvl(x, y)")
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("day_itvl(x, y)")
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True))

    x = pd.Series(["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"],
                  dtype="M8[s]")
    y = pd.Series(["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"],
                  dtype="M8[s]")
    ret = mon_itvl(x, y)
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True))
    ret = day_itvl(x, y)
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True))

    env = pd.DataFrame(dict(x=x, y=y))
    ret = envp.bind_env(env).parse("mon_itvl(x, y)")
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("day_itvl(x, y)")
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True))

    # Empty sequence.
    ret = mon_itvl([], [])
    assert len(ret) == 0
    ret = day_itvl([], [])
    assert len(ret) == 0

    ret = envp.bind_env(env).parse("mon_itvl([], [])")
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("day_itvl([], [])")
    assert len(ret) == 0


# %%
def test_busiday():
    envp = EnvParser()

    # List of string, np.ndarray and pd.Series.
    x = ["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"]
    ret = is_busiday(x)
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = not_busiday(x)
    assert np.all(ret == [0, 1, 1, 1, 0])

    env = dict(x=x)
    ret = envp.bind_env(env).parse("is_busiday(x)")
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = envp.bind_env(env).parse("not_busiday(x)")
    assert np.all(ret == [0, 1, 1, 1, 0])

    x = np.array(["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"])
    ret = is_busiday(x)
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = not_busiday(x)
    assert np.all(ret == [0, 1, 1, 1, 0])

    env = dict(x=x)
    ret = envp.bind_env(env).parse("is_busiday(x)")
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = envp.bind_env(env).parse("not_busiday(x)")
    assert np.all(ret == [0, 1, 1, 1, 0])

    x = pd.Series(["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"])
    ret = is_busiday(x)
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = not_busiday(x)
    assert np.all(ret == [0, 1, 1, 1, 0])

    env = pd.DataFrame(dict(x=x))
    ret = envp.bind_env(env).parse("is_busiday(x)")
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = envp.bind_env(env).parse("not_busiday(x)")
    assert np.all(ret == [0, 1, 1, 1, 0])

    # Single string and empty sequence.
    ret = is_busiday("2021-11-11")
    assert ret
    ret = is_busiday([])
    assert len(ret) == 0

    ret = envp.bind_env(env).parse('is_busiday("2021-11-11")')
    assert ret
    ret = envp.bind_env(env).parse("is_busiday([])")
    assert len(ret) == 0


# %%
def test_gethour():
    envp = EnvParser()

    # List of string, np.ndarray and pd.Series.
    x = ["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"]
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    env = dict(x=x)
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    x = np.array(["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"])
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    env = dict(x=x)
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    x = np.array(["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"], dtype="M8[s]")
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    env = pd.DataFrame(dict(x=x))
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    x = pd.Series(["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"])
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    env = pd.DataFrame(dict(x=x))
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    # Empty sequence.
    ret = get_hour([])
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("get_hour([])")
    assert len(ret) == 0
