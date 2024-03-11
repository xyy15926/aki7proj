#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_exgine.py
#   Author: xyy15926
#   Created: 2024-02-01 10:07:31
#   Updated: 2024-03-11 10:37:09
#   Description:
# ---------------------------------------------------------

# %%
import pytest

if __name__ == "__main__":
    from importlib import reload
    from flagbear import fliper
    from suitbear import exgine
    from azkaban import pboc_conf
    reload(fliper)
    reload(exgine)
    reload(pboc_conf)

import numpy as np
import pandas as pd
import os
import json
from suitbear.exgine import parse_2df, parse_parts, parse_2stages
from suitbear.exgine import transform_part, agg_part, apply_3stages
from flagbear.fliper import extract_field
from azkaban.pboc_conf import gen_confs, MAPPERS, TRANS_CONF

MAPPERS_ = {k: {kk: vv[0] for kk, vv in v.items()} for k, v in MAPPERS.items()}

ASSETS = os.path.join(os.curdir, "assets")

DTYPE_DEFAULT = {
    "INT": np.nan,
    "FLO": np.nan,
    "VAR": "",
    "CHA": "",
    "DAT": pd.to_datetime,
}
DTYPE_USE_DEFAULT = {
    "INT": 1,
    "FLO": 1,
    "VAR": 0,
    "CHA": 0,
    "DAT": 0,
}

PBOC_JSON = os.path.join(ASSETS, "pboc_utf8.json")
PBOC_PARTS = os.path.join(ASSETS, "pboc_parts.csv")
PBOC_FIELDS = os.path.join(ASSETS, "pboc_fields.csv")


# %%
def test_parse2df():
    pboc = open(PBOC_JSON, "r").read()
    src = pd.Series({"xfy": pboc, "xfy2": pboc})
    pconfs = pd.read_csv(PBOC_PARTS)
    fconfs = pd.read_csv(PBOC_FIELDS)
    fconfs["default"] = fconfs["dtype"].str[:3].str.upper().map(DTYPE_DEFAULT)
    fconfs["use_default"] = fconfs["dtype"].str[:3].str.upper().map(DTYPE_USE_DEFAULT)

    confs = fconfs[fconfs["part"] == "pboc_basic_info"].copy()
    basic_info = parse_2df(src, confs, 0)
    assert not np.all(basic_info.isna())

    for part in ["pboc_mobile", "pboc_acc_info"]:
        confs = fconfs[fconfs["part"] == part].copy()
        confs["steps"] = (pconfs.loc[pconfs["part"] == part, "steps_0"].iloc[0]
                          + ":" + confs["steps"])
        df = parse_2df(src, confs, 1)
        assert not np.all(df.isna())


# %%
def test_parse_parts():
    pboc = open(PBOC_JSON, "r").read()
    src = pd.Series({"xfy": pboc, "xfy2": pboc})
    pconfs = pd.read_csv(PBOC_PARTS)
    fconfs = pd.read_csv(PBOC_FIELDS)
    fconfs["default"] = fconfs["dtype"].str[:3].str.upper().map(DTYPE_DEFAULT)
    fconfs["use_default"] = fconfs["dtype"].str[:3].str.upper().map(DTYPE_USE_DEFAULT)

    pconf = pconfs.iloc[0]
    psrc = parse_parts(src, pconf)
    assert not np.all(psrc.isna())

    pconf = pconfs[pconfs["level"] == 1].iloc[0]
    psrc = parse_parts(src, pconf)
    assert not np.all(psrc.isna())

    pconf = pconfs[pconfs["level"] == 2].iloc[0]
    psrc = parse_parts(src, pconf)
    assert not np.all(psrc.isna())
    part = pconf["part"]
    confs = fconfs[fconfs["part"] == part].copy()
    dest = parse_2df(psrc, confs, 0)
    assert not np.all(dest.isna())


# %%
def test_parse_2stages_sep():
    pboc = open(PBOC_JSON, "r").read()
    pboc2 = pboc.replace("2019101617463675115707", "2019101617463675115708")
    src = pd.Series({"xfy": pboc, "xfy2": pboc2})
    pconfs = pd.read_csv(PBOC_PARTS)
    fconfs = pd.read_csv(PBOC_FIELDS)
    fconfs["default"] = fconfs["dtype"].str[:3].str.upper().map(DTYPE_DEFAULT)
    fconfs["use_default"] = fconfs["dtype"].str[:3].str.upper().map(DTYPE_USE_DEFAULT)

    rets = {}
    for idx, pconf in pconfs.iterrows():
        part = pconf["part"]
        psrc = parse_parts(src, pconf)
        confs = fconfs[fconfs["part"] == part].copy()
        dfret = parse_2df(psrc, confs, 0)
        # Dumps into json-string for `.to_sql`.
        # dfret.loc[:, confs["key"][confs["dtype"] == "text"]] = (
        #     dfret[confs["key"][confs["dtype"] == "text"]].applymap(
        #         lambda x: json.dumps(x, ensure_ascii=False)))
        rets[part] = dfret

    return rets

    # xlw = pd.ExcelWriter("pboc.xlsx")
    # for idx, df in rets.items():
    #     df.to_excel(xlw, sheet_name=idx)
    # xlw.close()


def test_parse_2stages():
    pboc = open(PBOC_JSON, "r").read()
    pboc2 = pboc.replace("2019101617463675115707", "2019101617463675115708")
    src = pd.Series({"xfy": pboc, "xfy2": pboc2})
    pconfs = pd.read_csv(PBOC_PARTS)
    fconfs = pd.read_csv(PBOC_FIELDS)
    fconfs["default"] = fconfs["dtype"].str[:3].str.upper().map(DTYPE_DEFAULT)
    fconfs["use_default"] = fconfs["dtype"].str[:3].str.upper().map(DTYPE_USE_DEFAULT)
    rets = parse_2stages(src, pconfs, fconfs)

    return rets


# %%
def test_transform_part():
    rets = test_parse_2stages()
    df = rets["pboc_acc_info"].copy()
    conf = pd.DataFrame(TRANS_CONF["pboc_acc_info"],
                        columns=["key", "trans", "conds", "cmt"])
    tdf = transform_part(df, conf, MAPPERS_)

    assert np.all(tdf.columns.intersection(conf["key"])
                  == conf["key"].drop_duplicates())

    return tdf


# %%
def test_agg_part():
    df = test_transform_part()
    pconfs, aconfs = gen_confs()
    conf = aconfs[aconfs["part"] == "acc_no_cat_info"]
    adf = agg_part(df, conf, ["rid", "certno"], MAPPERS_)

    assert np.all(adf.columns == conf["key"])


# %%
def test_pboc_agg_conf(to_csv: int = 0):
    pconfs, aconfs = gen_confs()
    tconfs = (pd.concat([pd.DataFrame(val, columns=["key", "trans", "conds", "cmt"])
                        for val in TRANS_CONF.values()],
                        keys=TRANS_CONF.keys())
              .droplevel(1)
              .reset_index()
              .rename(columns={"index": "part"}))

    pconfs.loc[pconfs["level"] == 0, "join_key"] = "index"
    pconfs.loc[pconfs["level"] == 1, "join_key"] = "index,accid"

    if to_csv:
        import csv
        pconfs.to_csv("pboc_vars_parts.csv", encoding="gbk")
        tconfs.to_csv("pboc_vars_trans.csv", encoding="gbk")
        aconfs.to_csv("pboc_vars_aggs.csv", encoding="gbk")
        pd.concat([pd.DataFrame(v).T for v in MAPPERS.values()],
                  axis=0,
                  keys=MAPPERS.keys()).to_csv("pboc_vars_maps.csv",
                                              quoting=csv.QUOTE_NONNUMERIC,
                                              encoding="gbk")

    # Get the part conf for parsing.
    ppconfs = pd.read_csv(PBOC_PARTS)
    ppconfs.loc[ppconfs["level"] == 0, "join_key"] = "index"
    ppconfs.loc[ppconfs["part"] == "pboc_acc_info", "join_key"] = "index,PD01AI01"

    pconfs_ = pd.concat([ppconfs, pconfs], axis=0)

    return tconfs, pconfs_, aconfs


def test_apply_3stage():
    # Parse pboc.
    src = test_parse_2stages()
    for part, df in src.items():
        df.index.set_names(["index"] + df.index.names[1:], inplace=True)

    tconfs, pconfs_, aconfs = test_pboc_agg_conf()

    ret = apply_3stages(src, tconfs, pconfs_, aconfs, MAPPERS_)

    return ret
