#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_exdf.py
#   Author: xyy15926
#   Created: 2025-06-10 19:29:26
#   Updated: 2025-07-29 16:28:02
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd
from importlib import reload
if __name__ == "__main__":
    from ubears.suitbear.dirt import exdf
    reload(exdf)
from ubears.suitbear.dirt.exdf import (
    flat_ft_dfs,
    trans_from_dfs,
    agg_from_dfs,
    agg_from_graphdf,
    dep_from_fconfs
)
from ubears.suitbear.kgraph.gxgine import GRAPH_NODE, GRAPH_REL


# %%
def pboc_rec():
    src = {
        "PRH": {
            "PA01": {
                "PA01A": {
                    "PA01AI01": "2019101617463675115707",
                    "PA01AR01": "2019-10-16T17:46:36"
                },
                "PA01B": {
                    "PA01BD01": "10",
                    "PA01BD02": "22",
                    "PA01BI01": "622926198501293785",
                    "PA01BI02": "A10311000H0001",
                    "PA01BQ01": "王小二"

                },
                "PA01C": None,
                "PA01D": None,
                "PA01E": {
                    "PA01ES01": "1"
                }
            }
        },
        "PDA": {
            "PD01": [
                {
                    "PD01A": {
                        "PD01AD01": "D1",
                        "PD01AD02": "12",
                        "PD01AD03": "12",
                        "PD01AD04": "JPY",
                        "PD01AI01": "92719",
                        "PD01AJ01": "10000",
                        "PD01AJ03": None,
                        "PD01AR01": "2013-01-10",
                        "PD01AR02": None,
                    },
                    "PD01B": {
                        "PD01BD01": "3",
                        "PD01BD04": "",
                        "PD01BJ01": "0",
                        "PD01BJ02": "",
                        "PD01BR01": "2018-06-20",
                        "PD01BR02": "--",
                    },
                    "PD01C": None,
                    "PD01D": {
                        "PD01DH": [
                            {
                                "PD01DD01": "#",
                                "PD01DR03": "2016-08"
                            },
                            {
                                "PD01DD01": "#",
                                "PD01DR03": "2016-07"
                            }
                        ],
                        "PD01DR01": "2016-07",
                        "PD01DR02": "2018-06"
                    },
                    "PD01E": {
                        "PD01EH": [
                            {
                                "PD01ED01": "#",
                                "PD01EJ01": "",
                                "PD01ER03": "2014-12"
                            },
                            {
                                "PD01ED01": "#",
                                "PD01EJ01": "",
                                "PD01ER03": "2014-11"
                            },
                            {
                                "PD01ED01": "#",
                                "PD01EJ01": "",
                                "PD01ER03": "2014-10"
                            }
                        ],
                        "PD01ER01": "2014-11",
                        "PD01ER02": "2018-06",
                        "PD01ES01": "44"
                    },
                    "PD01F": None,
                    "PD01G": None,
                    "PD01H": None,
                    "PD01Z": None,
                },
                {
                    "PD01A": {
                        "PD01AD01": "R4",
                        "PD01AD02": "12",
                        "PD01AD03": "92",
                        "PD01AD04": "CNY",
                        "PD01AI01": "92720",
                        "PD01AJ01": "20000",
                        "PD01AJ03": None,
                        "PD01AR01": "2013-01-10",
                        "PD01AR02": None,
                    },
                    "PD01B": {
                        "PD01BD01": "3",
                        "PD01BD04": "",
                        "PD01BJ01": "0",
                        "PD01BJ02": "",
                        "PD01BR01": "2018-06-20",
                        "PD01BR02": "--",
                    },
                    "PD01C": None,
                    "PD01D": {
                        "PD01DH": [
                            {
                                "PD01DD01": "#",
                                "PD01DR03": "2016-08"
                            },
                            {
                                "PD01DD01": "#",
                                "PD01DR03": "2016-07"
                            }
                        ],
                        "PD01DR01": "2016-07",
                        "PD01DR02": "2018-06"
                    },
                    "PD01E": None,
                    "PD01F": None,
                    "PD01G": None,
                    "PD01H": None,
                    "PD01Z": None,
                },
            ],
        }
    }

    return src


# %%
def pboc_flat_confs():
    flat_pconfs = pd.DataFrame.from_records([{
        "part": "acc_info",
        "level": 1,
        "prikey": ["rid", "certno"],
        "steps": [
            {
                "content": "PDA:PD01:[_]:PD01A",
                "key": [
                    ("rid", "PRH:PA01:PA01A:PA01AI01"),
                    ("certno", "PRH:PA01:PA01B:PA01BI01"),
                    ("accid", "PDA:PD01:[_]:PD01A:PD01AI01"),
                ]
            },
        ],
        "desc": "acc_info",
    }, ])
    flat_fconfs = pd.DataFrame([
        ["PD01AD01", "PD01AD01", "VARCHAR(31)", "基本信息_账户类型"],
        ["PD01AD02", "PD01AD02", "VARCHAR(31)", "基本信息_业务管理机构类型"],
        ["PD01AD03", "PD01AD03", "VARCHAR(31)", "基本信息_业务种类"],
        ["PD01AD04", "PD01AD04", "VARCHAR(31)", "基本信息_币种"],
        ["PD01AI01", "PD01AI01", "VARCHAR(255)", "基本信息_账户编号"],
        ["PD01AR01", "PD01AR01", "DATE", "基本信息_开立日期"],
        ["PD01AR02", "PD01AR02", "DATE", "基本信息_到期日期"],
    ], columns=["key", "step", "dtype", "desc"])
    flat_fconfs["part"] = "acc_info"

    return flat_pconfs, flat_fconfs


# %%
def test_flat_ft_dfs():
    # Flat and convert to determined dtypes for list of records.
    src = [pboc_rec(), ]
    flat_pconfs, flat_fconfs = pboc_flat_confs()
    part_name = flat_pconfs.loc[0, "part"]
    flat_rets = flat_ft_dfs(src, flat_pconfs, flat_fconfs)
    ret = flat_rets[part_name]
    assert np.all(ret.dtypes.value_counts().values == [5, 2])
    assert np.all(ret.values[:, :-1] == [
        ["D1", "12", "12", "JPY", "92719", np.datetime64("2013-01-10")],
        ["R4", "12", "92", "CNY", "92720", np.datetime64("2013-01-10")]
    ])
    assert np.all(np.isnat(ret["PD01AR02"]))

    # Check and convert to determined dtypes for dict of DFs.
    str_ret = {part_name: ret.astype(object)}
    flat_rets = flat_ft_dfs(str_ret, flat_pconfs, flat_fconfs)
    ret = flat_rets[part_name]
    assert np.all(ret.dtypes.value_counts().values == [5, 2])
    assert np.all(ret.values[:, :-1] == [
        ["D1", "12", "12", "JPY", "92719", np.datetime64("2013-01-10")],
        ["R4", "12", "92", "CNY", "92720", np.datetime64("2013-01-10")]
    ])
    assert np.all(np.isnat(ret["PD01AR02"]))


# %%
def pboc_trans_confs():
    trans_pconfs = pd.DataFrame.from_records([{
        "part": "acc_info",
        "desc": "分帐户明细",
        "level": 1,
        "prikey": ["rid", "certno", "accid"],
        "from_": ["acc_info"],
        "joinkey": None,
    }, ])
    trans_fconfs = pd.DataFrame([
        ["acc_moi_start"    , "mon_itvl(PD01AR01, today)", None              , "开立距今"],
        ["r4acc_moi_start"  , "mon_itvl(PD01AR01, today)", "PD01AD01==\"R4\"", "R4开立距今"],
    ], columns=["key", "trans", "cond", "desc"])
    trans_fconfs["part"] = "acc_info"

    return trans_pconfs, trans_fconfs


# %%
def test_trans_from_dfs():
    src = [pboc_rec(), ]
    flat_pconfs, flat_fconfs = pboc_flat_confs()
    part_name = flat_pconfs.loc[0, "part"]
    flat_rets = flat_ft_dfs(src, flat_pconfs, flat_fconfs)
    trans_pconfs, trans_fconfs = pboc_trans_confs()

    # Make a copy and the original value won't be modified.
    trans_rets_copy = trans_from_dfs(flat_rets, trans_pconfs, trans_fconfs, how="new")
    ret = flat_rets[part_name]
    assert np.all(ret.dtypes.value_counts().values == [5, 2])
    assert np.all(ret.values[:, :-1] == [
        ["D1", "12", "12", "JPY", "92719", np.datetime64("2013-01-10")],
        ["R4", "12", "92", "CNY", "92720", np.datetime64("2013-01-10")]
    ])
    mgap = (np.datetime64("today", "M") - np.datetime64("2013-01")).astype(float)
    assert np.all(trans_rets_copy[part_name]["acc_moi_start"] == -mgap)

    # Update the original DFs directly.
    trans_rets = trans_from_dfs(flat_rets, trans_pconfs, trans_fconfs)
    assert trans_rets[part_name] is flat_rets[part_name]
    assert np.all(trans_rets[part_name]["acc_moi_start"] == -mgap)


# %%
def pboc_agg_confs():
    agg_pconfs = pd.DataFrame.from_records([{
        "part": "acc_info_agg",
        "desc": "报文账户信息",
        "level": 0,
        "prikey": ["rid", "certno"],
        "from_": ["acc_info", "NEW"],
        "joinkey": [["rid", "certno", "accid"],
                    ["rid", "certno", "accid"]],
    }, ])
    agg_fconfs = pd.DataFrame([
        ["acc_ms_max"   , "max(-acc_moi_start)"     , None                  , "距今最大"],
        ["r4acc_ms_max1", "max(-acc_moi_start)"     , "PD01AD01==\"R4\""    , "R4距今最大"],
        ["r4acc_ms_max2", "max(-r4acc_moi_start)"   , None                  , "R4距今最大"],
    ], columns=["key", "agg", "cond", "desc"])
    agg_fconfs["part"] = "acc_info_agg"

    return agg_pconfs, agg_fconfs


# %%
def test_agg_on_dfs():
    src = [pboc_rec(), ]
    flat_pconfs, flat_fconfs = pboc_flat_confs()
    part_name = flat_pconfs.loc[0, "part"]
    flat_rets = flat_ft_dfs(src, flat_pconfs, flat_fconfs)
    trans_pconfs, trans_fconfs = pboc_trans_confs()
    trans_rets_copy = trans_from_dfs(flat_rets, trans_pconfs, trans_fconfs, how="new")
    flat_rets["NEW"] = trans_rets_copy[part_name]

    agg_pconfs, agg_fconfs = pboc_agg_confs()
    agg_rets = agg_from_dfs(flat_rets, agg_pconfs, agg_fconfs)
    agg_pname = agg_pconfs.loc[0, "part"]
    mgap = (np.datetime64("today", "M") - np.datetime64("2013-01")).astype(float)
    assert np.all(agg_rets[agg_pname] == mgap)


# %%
def test_dep_from_fconfs():
    flat_pconfs, flat_fconfs = pboc_flat_confs()
    trans_pconfs, trans_fconfs = pboc_trans_confs()
    agg_pconfs, agg_fconfs = pboc_agg_confs()
    pconfs = pd.concat([flat_pconfs, trans_pconfs, agg_pconfs])
    fconfs = pd.concat([flat_fconfs, trans_fconfs, agg_fconfs])
    target = ["r4acc_ms_max1"]
    deps = dep_from_fconfs(target, fconfs)
    assert deps["key"].tolist() == ["PD01AD01", "PD01AR01",
                                    "acc_moi_start", "r4acc_ms_max1"]
