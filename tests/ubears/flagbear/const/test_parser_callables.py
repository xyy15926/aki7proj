#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_parser_callables.py
#   Author: xyy15926
#   Created: 2025-02-21 14:11:51
#   Updated: 2025-02-21 18:05:39
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
from ubears.flagbear.llp.parser import EnvParser


# %% -------------------------------------------------------------------------
#                    * * * Aggregation Callables * * *
# %% -------------------------------------------------------------------------
# %%
def test_nnfilter():
    envp = EnvParser()

    env = dict(x=[1, 2, 2, None])
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert np.all(ret == [1, 2, 2])

    env = dict(x=np.array([1, 2, 2, None]))
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert np.all(ret == [1, 2, 2])

    env = dict(x=pd.Series([1, 2, 2, None]))
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert np.all(np.isclose(ret, [1, 2, 2, np.nan], equal_nan=True))

    env = dict(x=[])
    ret = envp.bind_env(env).parse("nnfilter(x)")
    assert ret == []


# %%
def test_getn():
    envp = EnvParser()

    env = dict(x=[1, 2, 3, 4])
    ret = envp.bind_env(env).parse("getn(x, 1)")
    assert ret == 2
    ret = envp.bind_env(env).parse("getn(x, 7)")
    assert ret is None
    ret = envp.bind_env(env).parse("getn(x, None)")
    assert ret is None

    env = dict(x=np.array([1, 2, 3, 4]))
    ret = envp.bind_env(env).parse("getn(x, 1)")
    assert ret == 2
    ret = envp.bind_env(env).parse("getn(x, 7)")
    assert ret is None
    ret = envp.bind_env(env).parse("getn(x, None)")
    assert ret is None

    env = dict(x=pd.Series([1, 2, 3, 4]))
    ret = envp.bind_env(env).parse("getn(x, 1)")
    assert ret == 2
    ret = envp.bind_env(env).parse("getn(x, 7)")
    assert ret is None
    ret = envp.bind_env(env).parse("getn(x, None)")
    assert ret is None

    ret = envp.bind_env(env).parse("getn([], 1)")
    assert ret is None


# %%
def test_argmaxmin():
    envp = EnvParser()

    env = dict(x=[1, 1, 2, 2, np.nan])
    ret = envp.bind_env(env).parse("argmax(x)")
    assert ret == 2

    env = dict(x=np.array([1, 1, 2, 2, np.nan]))
    ret = envp.bind_env(env).parse("argmax(x)")
    assert ret == 2

    env = dict(x=pd.Series([1, 1, 2, 2, np.nan]))
    ret = envp.bind_env(env).parse("argmax(x)")
    assert ret == 2

    ret = envp.bind_env(env).parse("argmax([])")
    assert ret is None


# %%
def test_argmaxsmins():
    envp = EnvParser()

    env = dict(x=[1, 1, 2, 2, np.nan],
               y=[1, 2, 3, 4, 5])
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == [3, 4])
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [1, 2])

    env = dict(x=np.array([1, 1, 2, 2, np.nan]),
               y=np.array([1, 2, 3, 4, 5]))
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == [3, 4])
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [1, 2])

    env = pd.DataFrame(dict(x=[1, 1, 2, 2, np.nan],
                            y=[1, 2, 3, 4, 5]))
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == [3, 4])
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [1, 2])

    env = dict(x=np.array([np.nan] * 5),
               y=np.array([1, 2, 3, 4, 5]))
    with pytest.warns(RuntimeWarning):
        ret = envp.bind_env(env).parse("argmaxs(x, y)")
        assert len(ret) == 0

    x = np.array(["2021-12-12",
                  "2021-11-11",
                  "2022-11-11",
                  "2021-11-11",
                  "NaT"], dtype="M8[D]")
    y = [1, 2, 3, 4, 5]
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
    env = pd.DataFrame(dict(x=x, y=y))
    ret = envp.bind_env(env).parse("argmaxs(x, y)")
    assert np.all(ret == 3)
    ret = envp.bind_env(env).parse("argmins(x, y)")
    assert np.all(ret == [2, 4])

    ret = envp.bind_env(env).parse("argmaxs([], [])")
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("argmins([], [])")
    assert len(ret) == 0


# %%
def test_cals():
    envp = EnvParser()

    env = dict(x=[1, 1, 1, 1, 0, 4, 5, 0])
    ret = envp.bind_env(env).parse("flat1_max(x)")
    assert ret == 4
    ret = envp.bind_env(env).parse("flat1_max([])")
    assert ret == 0
    ret = envp.bind_env(env).parse("flat1_max([5])")
    assert ret == 1
    ret = envp.bind_env(env).parse("flat1_max([0])")
    assert ret == 0

    env = dict(x=[1, 2, 1, 2, 0])
    ret = envp.bind_env(env).parse("coef_var(x)")
    assert ret == np.std(env["x"]) / np.mean(env["x"])
    ret = envp.bind_env(env).parse("coef_var([])")
    assert ret == 0


# %% -------------------------------------------------------------------------
#                    * * * Transformation Callables * * *
# %% -------------------------------------------------------------------------
def test_sortby():
    envp = EnvParser()

    env = dict(x=[11, np.nan, 13, 4, np.nan],
               y=[1, 12, 3, 14, np.nan])
    ret = envp.bind_env(env).parse("sortby(x, y)")
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan],
                             equal_nan=True))

    env = dict(x=np.array([11, np.nan, 13, 4, np.nan]),
               y=np.array([1, 12, 3, 14, np.nan]))
    ret = envp.bind_env(env).parse("sortby(x, y)")
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan],
                             equal_nan=True))

    env = pd.DataFrame({"x": pd.Series([11, np.nan, 13, 4, np.nan]),
                        "y": pd.Series([1, 12, 3, 14, np.nan])})
    ret = envp.bind_env(env).parse("sortby(x, y)")
    assert np.all(np.isclose(ret, [11, 13, np.nan, 4, np.nan],
                             equal_nan=True))


# %%
def test_maps():
    envp = EnvParser()
    ref = dict(zip("abcde", range(1, 6)))

    x = ["a", "c", "d", "e", "b", None, "NA"]
    env = dict(x=x, ref=ref)
    ret = envp.bind_env(env).parse("map(x, ref)")
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = envp.bind_env(env).parse("map(x, ref, nan)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan],
                             equal_nan=True))

    x = np.array(["a", "c", "d", "e", "b", None, "NA"])
    env = dict(x=x, ref=ref)
    ret = envp.bind_env(env).parse("map(x, ref)")
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = envp.bind_env(env).parse("map(x, ref, nan)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan],
                             equal_nan=True))

    x = pd.Series(["a", "c", "d", "e", "b", None, "NA"])
    env = dict(x=x, ref=ref)
    ret = envp.bind_env(env).parse("map(x, ref)")
    assert np.all(ret == [1, 3, 4, 5, 2, None, None])
    ret = envp.bind_env(env).parse("map(x, ref, nan)")
    assert np.all(np.isclose(ret, [1, 3, 4, 5, 2, np.nan, np.nan],
                             equal_nan=True))

    x = ["a,b,c,d,e,na,g", "a,g,c", "g"]
    env = dict(x=x, ref=ref)
    ret = envp.bind_env(env).parse("sep_map(x, ref)")
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = envp.bind_env(env).parse('sep_map(x, ref, ",", ":")')
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])

    x = np.array(["a,b,c,d,e,na,g", "a,g,c", "g"])
    env = dict(x=x, ref=ref)
    ret = envp.bind_env(env).parse("sep_map(x, ref)")
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = envp.bind_env(env).parse('sep_map(x, ref, ",", ":")')
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])

    x = pd.Series(["a,b,c,d,e,na,g", "a,g,c", "g"])
    ret = envp.bind_env(env).parse("sep_map(x, ref)")
    for le, re in zip(ret, [(1, 2, 3, 4, 5), (1, 3), ()]):
        assert le == re
    ret = envp.bind_env(env).parse('sep_map(x, ref, ",", ":")')
    assert np.all(ret == ["1:2:3:4:5", "1:3", ""])


# %%
def test_cb_fstmaxmin():
    envp = EnvParser()

    x = [11, np.nan, 13, 4, np.nan]
    y = [1, 12, 3, 14, np.nan]
    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(np.isclose(ret, [1, 12, 3, 4, np.nan], equal_nan=True))

    x = np.array([11, np.nan, 13, 4, np.nan])
    y = np.array([1, 12, 3, 14, np.nan])
    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_min(x, y)")
    assert np.all(np.isclose(ret, [1, 12, 3, 4, np.nan], equal_nan=True))

    env = pd.DataFrame({"x": [11, np.nan, 13, 4, np.nan],
                        "y": [1, 12, 3, 14, np.nan]})
    ret = envp.bind_env(env).parse("cb_fst(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 4, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_max(x, y)")
    assert np.all(np.isclose(ret, [11, 12, 13, 14, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("cb_min(x, y)")
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


# %%
def test_mon_day_itvl():
    envp = EnvParser()

    x = ["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"]
    y = ["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"]
    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("mon_itvl(x, y)")
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan],
                             equal_nan=True))
    ret = envp.bind_env(env).parse("day_itvl(x, y)")
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan],
                             equal_nan=True))

    x = np.array(["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"],
                 dtype="M8[D]")
    y = np.array(["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"],
                 dtype="M8[D]")
    env = dict(x=x, y=y)
    ret = envp.bind_env(env).parse("mon_itvl(x, y)")
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan],
                             equal_nan=True))
    ret = envp.bind_env(env).parse("day_itvl(x, y)")
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan],
                             equal_nan=True))

    env = pd.DataFrame({
        "x": np.array(["2021-11-11", "NaT", "2021-11-13", "2021-11-14", "NaT"],
                      dtype="M8[D]"),
        "y": ["2021-12-01", "2021-12-31", "2021-12-31", "2021-12-02", "NaT"]})
    # Pandas will cast np.M8[D] into M8[s] or M8[ns] depending on versions.
    assert env.dtypes.iloc[0] != "object"
    assert env.dtypes.iloc[1] == "object"
    ret = envp.bind_env(env).parse("mon_itvl(x, y)")
    assert np.all(np.isclose(ret, [-1, np.nan, -1, -1, np.nan], equal_nan=True))
    ret = envp.bind_env(env).parse("day_itvl(x, y)")
    assert np.all(np.isclose(ret, [-20, np.nan, -48, -18, np.nan], equal_nan=True))

    ret = envp.bind_env(env).parse("mon_itvl([], [])")
    assert len(ret) == 0
    ret = envp.bind_env(env).parse("day_itvl([], [])")
    assert len(ret) == 0


# %%
def test_busiday():
    envp = EnvParser()

    env = dict(x=["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"])
    ret = envp.bind_env(env).parse("is_busiday(x)")
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = envp.bind_env(env).parse("not_busiday(x)")
    assert np.all(ret == [0, 1, 1, 1, 0])

    env = dict(x=np.array(["2021-11-11", "2025-01-01", "2021-11-13",
                           "2021-11-14", "NaT"]))
    ret = envp.bind_env(env).parse("is_busiday(x)")
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = envp.bind_env(env).parse("not_busiday(x)")
    assert np.all(ret == [0, 1, 1, 1, 0])

    env = dict(x=pd.Series(["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"]))
    ret = envp.bind_env(env).parse("is_busiday(x)")
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = envp.bind_env(env).parse("not_busiday(x)")
    assert np.all(ret == [0, 1, 1, 1, 0])

    ret = envp.bind_env(env).parse('is_busiday("2021-11-11")')
    assert ret
    ret = envp.bind_env(env).parse("is_busiday([])")
    assert len(ret) == 0


# %%
def test_gethour():
    envp = EnvParser()

    env = dict(x=["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"])
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    env = dict(x=np.array(["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"]))
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    env = dict(x=np.array(["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"],
                          dtype="M8[s]"))
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    env = dict(x=pd.Series(["2021-11-11T11:11:12", "2025-01-01T13:12:12", "NaT"]))
    ret = envp.bind_env(env).parse("get_hour(x)")
    assert np.all(np.isclose(ret, [11, 13, np.nan], equal_nan=True))

    ret = envp.bind_env(env).parse("get_hour([])")
    assert len(ret) == 0
