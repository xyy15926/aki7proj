#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_ex4df.py
#   Author: xyy15926
#   Created: 2024-04-15 18:17:58
#   Updated: 2024-12-14 23:22:05
#   Description:
# ---------------------------------------------------------

# %%
import pytest

if __name__ == "__main__":
    from importlib import reload
    from flagbear.str2 import fliper
    from modsbear.dflater import ex2df, ex4df, exenv
    reload(fliper)
    reload(ex2df)
    reload(ex4df)
    reload(exenv)

import numpy as np
import pandas as pd
import os
import json

from flagbear.slp.finer import get_assets_path
from flagbear.str2.fliper import extract_field, rebuild_dict
from modsbear.dflater.ex2df import rebuild_rec2df
from modsbear.dflater.ex4df import agg_on_df, trans_on_df


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
def pboc_acc_info():
    rec = pboc_rec()
    val_rules = [
        ["pboc_acc_info", "PDA:PD01:[_]"],
    ]
    index_rules = [
        ["rid", "PRH:PA01:PA01A:PA01AI01"],
        ["certno", "PRH:PA01:PA01B:PA01BI01"],
    ]
    nrec = rebuild_rec2df(rec, val_rules, index_rules, explode=True)

    fval_rules = [
        ["PD01AD01", "PD01A:PD01AD01", "VARCHAR(31)",],
        ["PD01AD02", "PD01A:PD01AD02", "VARCHAR(31)",],
        ["PD01AD03", "PD01A:PD01AD03", "VARCHAR(31)",],
        ["PD01AD04", "PD01A:PD01AD04", "VARCHAR(31)",],
    ]
    findex_rules = [
        ["accid", "PD01A:PD01AI01"]
    ]
    fields = nrec["pboc_acc_info"].apply(rebuild_rec2df,
                                         val_rules=fval_rules,
                                         index_rules=findex_rules,
                                         explode=True)
    fields = pd.concat(fields.values, keys=nrec.index)

    return fields


# %%
def test_trans_on_df():
    src = pboc_acc_info()
    mapper = {
        "cdr_cat": {
            "D1": (1        , "非循环贷账户"),
            "R1": (3        , "循环贷账户"),
            "R2": (4        , "贷记卡账户"),
            "R3": (5        , "准贷记卡账户"),
            "R4": (2        , "循环额度下分账户"),
            "C1": (99       , "催收账户"),
        },
        "exchange_rate": {
            "USD": (7       , "USD"),
            "EUR": (7.7     , "EUR"),
            "JPY": (0.05    , "JPY"),
            "CNY": (1       , "CNY"),
            "AUD": (4.7     , "AUD"),
            "RUB": (0.07    , "RUB"),
            "CAD": (5.3     , "CAD"),
        },
    }
    mapper = {k: {kk: vv[0] for kk, vv in v.items()} for k,v in mapper.items()}

    trans_rules = [
        ["acc_cat", "map(PD01AD01, cdr_cat)"],
        ["acc_exchange_rate", "acc_cat != 99", "map(PD01AD04, exchange_rate)"],
    ]
    transed = trans_on_df(src, trans_rules, env=mapper)
    assert np.all(transed.loc[transed["acc_cat"] == 99, "acc_exchange_rate"].isna())
    assert np.all(transed.loc[transed["acc_cat"] != 99, "acc_exchange_rate"].notna())
    mret = {k: v for i, k, v in transed[["PD01AD01", "acc_cat"]].itertuples()}
    for k, v in mret.items():
        assert v == mapper["cdr_cat"][k]

    return transed


# %%
def test_agg_on_df():
    src = test_trans_on_df()
    agg_rules = [
        ["c1_acc_cat_cnt", "acc_cat == 99", "count(_)"],
        ["d1r41_acc_cat_cnt", "acc_cat <=3", "count(_)"],
        ["d1r4_acc_cat_cnt", "(acc_cat == 1) | (acc_cat == 2)", "count(_)"],
        ["cnt", None, "count(_)"],
        ["cnt2", "count(_)"],
    ]
    agged = agg_on_df(src, agg_rules)
    assert agged["cnt"] == len(src)
    assert agged["cnt2"] == len(src)
    assert agged["c1_acc_cat_cnt"] == (src["acc_cat"] == 99).sum()
    assert agged["d1r41_acc_cat_cnt"] == (src["acc_cat"] <= 3).sum()
    assert agged["d1r4_acc_cat_cnt"] == (src["acc_cat"] <= 2).sum()
