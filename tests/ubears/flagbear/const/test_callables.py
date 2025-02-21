#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_callables.py
#   Author: xyy15926arruns
#   Created: 2025-02-20 18:56:36
#   Updated: 2025-02-21 11:31:31
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ubears.flagbear.const import callables
    from ubears.modsbear.locale import calender
    reload(callables)
    reload(calender)

import numpy as np
import pandas as pd
from ubears.flagbear.const.callables import (
    # Aggregation
    nnfilter, getn,
    argmax, argmin, argmaxs, argmins,
    flat1_max, coef_var,
    # Transformation
    sortby,
    map, sep_map,
    cb_fst, cb_max, cb_min,
    mon_itvl, day_itvl,
    is_busiday, not_busiday, get_hour,
)


# %% -------------------------------------------------------------------------
#                    * * * Aggregation Callables * * *
# %% -------------------------------------------------------------------------
# %%
def test_nnfilter():
    x = [1, 2, 2, None]
    ret = nnfilter(x)
    assert np.all(ret == [1, 2, 2])

    x = np.array([1, 2, 2, None])
    ret = nnfilter(x)
    assert np.all(ret == [1, 2, 2])

    x = pd.Series([1, 2, 2, None])
    ret = nnfilter(x)
    assert np.all(np.isclose(ret, [1, 2, 2, np.nan], equal_nan=True))

    ret = nnfilter([])
    assert ret == []


# %%
def test_getn():
    x = [1, 2, 3, 4]
    ret = getn(x, 1)
    assert ret == 2
    ret = getn(x, 7)
    assert ret is None
    ret = getn(x, None)
    assert ret is None

    x = np.array([1, 2, 3, 4])
    ret = getn(x, 1)
    assert ret == 2
    ret = getn(x, 7)
    assert ret is None
    ret = getn(x, None)
    assert ret is None

    x = pd.Series([1, 2, 3, 4])
    ret = getn(x, 1)
    assert ret == 2
    ret = getn(x, 7)
    assert ret is None
    ret = getn(x, None)
    assert ret is None

    ret = getn([], 1)
    assert ret is None


# %%
def test_argmaxmin():
    x = [1, 1, 2, 2, np.nan]
    ret = argmax(x)
    assert ret == 2

    x = np.array([1, 1, 2, 2, np.nan])
    ret = argmax(x)
    assert ret == 2

    x = pd.Series([1, 1, 2, 2, np.nan])
    ret = argmax(x)
    assert ret == 2

    ret = argmax([])
    assert ret is None


# %%
def test_argmaxsmins():
    x = [1, 1, 2, 2, np.nan]
    y = [1, 2, 3, 4, 5]
    ret = argmaxs(x, y)
    assert np.all(ret == [3, 4])
    ret = argmins(x, y)
    assert np.all(ret == [1, 2])

    x = np.array([1, 1, 2, 2, np.nan])
    y = np.array([1, 2, 3, 4, 5])
    ret = argmaxs(x, y)
    assert np.all(ret == [3, 4])
    ret = argmins(x, y)
    assert np.all(ret == [1, 2])

    x = pd.Series([1, 1, 2, 2, np.nan])
    y = pd.Series([1, 2, 3, 4, 5])
    ret = argmaxs(x, y)
    assert np.all(ret == [3, 4])
    ret = argmins(x, y)
    assert np.all(ret == [1, 2])

    x = np.array([np.nan] * 5)
    y = np.array([1, 2, 3, 4, 5])
    with pytest.warns(RuntimeWarning):
        ret = argmaxs(x, y)
        assert len(ret) == 0

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

    ret = argmaxs([], [])
    assert len(ret) == 0
    ret = argmins([], [])
    assert len(ret) == 0


# %%
def test_cals():
    x = [1, 1, 1, 1, 0, 4, 5, 0]
    ret = flat1_max(x)
    assert ret == 4
    ret = flat1_max([])
    assert ret == 0
    ret = flat1_max([5])
    assert ret == 1
    ret = flat1_max([0])
    assert ret == 0

    x = [1, 2, 1, 2, 0]
    ret = coef_var(x)
    assert ret == np.std(x) / np.mean(x)
    ret = coef_var([])
    assert ret == 0


# %% -------------------------------------------------------------------------
#                    * * * Transformation Callables * * *
# %% -------------------------------------------------------------------------
def test_sortby():
    x = [11, np.nan, 13, 4, np.nan]
    y = [1, 12, 3, 14, np.nan]
    ret = sortby(x, y)
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    x = np.array([11, np.nan, 13, 4, np.nan])
    y = np.array([1, 12, 3, 14, np.nan])
    ret = sortby(x, y)
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))

    x = pd.Series([11, np.nan, 13, 4, np.nan])
    y = pd.Series([1, 12, 3, 14, np.nan])
    ret = sortby(x, y)
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan], equal_nan=True))


# %%
def test_maps():
    ref = dict(zip("abcde", range(1, 6)))

    x = ["a", "c", "d", "e", "b", None, "NA"]
    ret = map(x, ref)
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = map(x, ref, np.nan)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True))

    x = np.array(["a", "c", "d", "e", "b", None, "NA"])
    ret = map(x, ref)
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = map(x, ref, np.nan)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True))

    x = pd.Series(["a", "c", "d", "e", "b", None, "NA"])
    ret = map(x, ref)
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = map(x, ref, np.nan)
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan], equal_nan=True))

    x = ["a,b,c,d,e,na,g", "a,g,c", "g"]
    ret = sep_map(x, ref)
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = sep_map(x, ref, ",", ":")
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])

    x = np.array(["a,b,c,d,e,na,g", "a,g,c", "g"])
    ret = sep_map(x, ref)
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = sep_map(x, ref, ",", ":")
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])

    x = pd.Series(["a,b,c,d,e,na,g", "a,g,c", "g"])
    ret = sep_map(x, ref)
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = sep_map(x, ref, ",", ":")
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])


# %%
def test_cb_fstmaxmin():
    x = [11, np.nan, 13, 4, np.nan]
    y = [1, 12, 3, 14, np.nan]
    ret = cb_fst(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = cb_max(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = cb_min(x, y)
    assert np.all(np.isclose(ret, [1, 12, 3, 4, np.nan], equal_nan=True))

    x = np.array([11, np.nan, 13, 4, np.nan])
    y = np.array([1, 12, 3, 14, np.nan])
    ret = cb_fst(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = cb_max(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = cb_min(x, y)
    assert np.all(np.isclose(ret, [1, 12, 3, 4, np.nan], equal_nan=True))

    x = pd.Series([11, np.nan, 13, 4, np.nan])
    y = pd.Series([1, 12, 3, 14, np.nan])
    ret = cb_fst(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = cb_max(x, y)
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = cb_min(x, y)
    assert np.all(np.isclose(ret, [1, 12, 3, 4, np.nan], equal_nan=True))

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


# %%
def test_mon_day_itvl():
    x = ["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"]
    y = ["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"]
    ret = mon_itvl(x, y)
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True))
    ret = day_itvl(x, y)
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True))

    x = np.array(["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"],
                 dtype="M8[D]")
    y = np.array(["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"],
                 dtype="M8[D]")
    ret = mon_itvl(x, y)
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True))
    ret = day_itvl(x, y)
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True))

    x = pd.Series(["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"],
                  dtype="M8[s]")
    y = pd.Series(["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"],
                  dtype="M8[s]")
    ret = mon_itvl(x, y)
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True))
    ret = day_itvl(x, y)
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True))

    ret = mon_itvl([], [])
    assert len(ret) == 0
    ret = day_itvl([], [])
    assert len(ret) == 0


# %%
def test_busiday():
    x = ["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"]
    ret = is_busiday(x)
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = not_busiday(x)
    assert np.all(ret == [0, 1, 1, 1, 0])

    x = np.array(["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"])
    ret = is_busiday(x)
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = not_busiday(x)
    assert np.all(ret == [0, 1, 1, 1, 0])

    x = pd.Series(["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"])
    ret = is_busiday(x)
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = not_busiday(x)
    assert np.all(ret == [0, 1, 1, 1, 0])

    ret = is_busiday("2021-11-11")
    assert ret
    ret = is_busiday([])
    assert len(ret) == 0


# %%
def test_gethour():
    x = ["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"]
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    x = np.array(["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"])
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    x = np.array(["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"], dtype="M8[s]")
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    x = pd.Series(["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"])
    ret = get_hour(x)
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    ret = get_hour([])
    assert len(ret) == 0
