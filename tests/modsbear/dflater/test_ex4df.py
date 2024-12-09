#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_ex4df.py
#   Author: xyy15926
#   Created: 2024-04-15 18:17:58
#   Updated: 2024-12-09 22:10:33
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

PBOC_CONFS = get_assets_path() / "pboc_settings"
PBOC_JSON = PBOC_CONFS / "pboc_utf8.json"


# %%
def pboc_src():
    pboc = open(PBOC_JSON, "r").read()
    pboc2 = pboc.replace("2019101617463675115707", "2019101617463675115708")
    src = pd.Series({"xfy": pboc, "xfy2": pboc2})

    return src


def pboc_acc_info():
    src = pboc_src()
    rec = src.iloc[0]
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
    assert ({k: v for i, k, v in transed[["PD01AD01", "acc_cat"]].itertuples()}
            == mapper["cdr_cat"])

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
