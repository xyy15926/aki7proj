#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_fxgine.py
#   Author: xyy15926
#   Created: 2024-04-19 14:58:22
#   Updated: 2024-05-17 21:11:20
#   Description:
# ---------------------------------------------------------

# %%
import pytest

if __name__ == "__main__":
    from importlib import reload
    from flagbear import fliper
    from modsbear import exgine
    from suitbear import fxgine
    reload(fliper)
    reload(exgine)
    reload(fxgine)

from functools import partial
import numpy as np
import pandas as pd
import os
import json
from flagbear.fliper import extract_field
from modsbear.exgine import rebuild_rec2df
from suitbear.fxgine import compress_hierarchy, flat_records
from suitbear.fxgine import agg_from_dfs
import logging
logging.basicConfig(level=logging.WARNING, force=True)

# %%
ASSETS = os.path.join(os.curdir, "assets")
PBOC_JSON = os.path.join(ASSETS, "pboc_utf8.json")
PBOC_PARTS = os.path.join(ASSETS, "pboc_parts.csv")
PBOC_FIELDS = os.path.join(ASSETS, "pboc_fields.csv")

MAPPER = {
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
    "repay_status": {
        "1": (1  , "逾期1-30天"),
        "2": (2  , "逾期31-60天"),
        "3": (3  , "逾期61-90天"),
        "4": (4  , "逾期91-120天"),
        "5": (5  , "逾期121-150天"),
        "6": (6  , "逾期151-180天"),
        "7": (7  , "逾期180天以上"),
        "*": (0  , "当月不需要还款且之前没有拖欠"),
        "#": (0  , "未知"),
        "/": (0  , "跳过"),
        "A": (0  , "账单日调整,当月不出单"),
        "B": (7  , "呆账"),
        "C": (0  , "结清、正常销户"),
        "D": (3  , "担保人代还"),
        "G": (7  , "（非正常销户）结束"),
        "M": (0  , "约定还款日后月底前还款"),
        "N": (0  , "正常还款"),
        "Z": (3  , "以资抵债"),
    },
}
MAPPER = {k: {kk: vv[0] for kk, vv in v.items()} for k,v in MAPPER.items()}


# %%
def pboc_src():
    pboc = open(PBOC_JSON, "r").read()
    pboc2 = pboc.replace("2019101617463675115707", "2019101617463675115708")
    src = pd.Series({"xfy": pboc, "xfy2": pboc2})

    return src


# %%
def test_compress_hierarchy():
    src = pboc_src()

    acc_info_part = {
        "part": "acc_info",
        "desc": "账户信息",
        "level": 1,
        "steps_0": "PDA:PD01:[_]:PD01A",
        "idkey_0": "PRH:PA01:PA01A:PA01AI01,PRH:PA01:PA01B:PA01BI01,PDA:PD01:[_]:PD01A:PD01AI01",
        "idname_0": "rid,certno,accid",
    }
    acc_info_psrc = compress_hierarchy(src, acc_info_part)
    assert len(acc_info_psrc) > len(src)
    assert acc_info_psrc.index.nlevels == src.index.nlevels + 2 + 1

    repay_60m_part = {
        "part": "acc_info",
        "desc": "近60个月还款",
        "level": 1,
        "steps_0": "PDA:PD01:[_]:PD01E",
        "idkey_0": "PRH:PA01:PA01A:PA01AI01,PRH:PA01:PA01B:PA01BI01,PDA:PD01:[_]:PD01A:PD01AI01",
        "idname_0": "rid,certno,accid",
        "steps_1": "PD01EH:[_]",
    }
    repay_60m_psrc = compress_hierarchy(src, repay_60m_part)
    assert repay_60m_psrc.index.nlevels == src.index.nlevels + 3 + 1
    assert len(repay_60m_psrc) > len(acc_info_psrc)

    repay_60m_part_v2 = {
        "part": "pboc_acc_repay_60_monthly",
        "desc": "分帐户明细_近60个月还款",
        "level": 2,
        "steps_0": "PDA:PD01:[_]",
        "steps_1": "PD01E:PD01EH:[_]",
    }
    repay_60m_psrc_v2 = compress_hierarchy(src, repay_60m_part_v2)
    assert repay_60m_psrc_v2.index.nlevels == src.index.nlevels + 1 + 1
    assert np.all(repay_60m_psrc.values == repay_60m_psrc_v2.values)

    return acc_info_psrc, repay_60m_psrc


def test_compress_hierarchy_range_idx():
    src = pboc_src()
    repay_60m_part = {
        "part": "acc_info",
        "desc": "近60个月还款",
        "level": 1,
        "steps_0": "PDA:PD01:[_]:PD01E",
        "idkey_0": "PRH:PA01:PA01A:PA01AI01,PRH:PA01:PA01B:PA01BI01,RANGEINDEX",
        "idname_0": "rid,certno,accid",
        "steps_1": "PD01EH:[_]",
    }
    repay_60m_psrc = compress_hierarchy(src, repay_60m_part)
    assert repay_60m_psrc.index.nlevels == src.index.nlevels + 3 + 1


def test_compress_hierarchy_null_vals():
    src = pboc_src()
    repay_60m_part = {
        "part": "acc_info",
        "desc": "近60个月还款",
        "level": 1,
        "steps_0": "PDA:PD01:[_]:PD01E",
        "idkey_0": "PRH:PA01:PA01A:PA01AI01,PRH:PA01:PA01B:PA01BI01,RANGEINDEX",
        "idname_0": "rid,certno,accid",
        "steps_1": "PD01AH:[_]",
    }
    repay_60m_psrc = compress_hierarchy(src, repay_60m_part)
    assert repay_60m_psrc.empty


# %%
def test_flat_record():
    acc_info_psrc, repay_60m_psrc = test_compress_hierarchy()

    acc_info_fields = pd.DataFrame([
        ["PD01AD01", "PD01AD01", "varchar(31)", "基本信息_账户类型"],
        ["PD01AD02", "PD01AD02", "varchar(31)", "基本信息_业务管理机构类型"],
        ["PD01AD03", "PD01AD03", "varchar(31)", "基本信息_业务种类"],
        ["PD01AD04", "PD01AD04", "varchar(31)", "基本信息_币种"],
        ["PD01AI01", "PD01AI01", "varchar(255)", "基本信息_账户编号"],
    ], columns=["key", "steps", "dtype", "desc"])
    acc_info_vals = flat_records(acc_info_psrc, acc_info_fields)
    assert np.all(acc_info_vals.index.names == acc_info_psrc.index.names)
    assert len(acc_info_vals) == len(acc_info_psrc)

    repay_60m_fields = pd.DataFrame.from_records([
        ("PD01ER03", "PD01ER03", "date", "月份"),
        ("PD01ED01", "PD01ED01", "varchar(31)", "还款状态"),
        ("PD01EJ01", "PD01EJ01", "int", "逾期（透支）总额"),
    ], columns=["key", "steps", "dtype", "desc"])
    repay_60m_vals = flat_records(repay_60m_psrc, repay_60m_fields, drop_rid=False)
    assert np.all(repay_60m_vals.index.names[:-1] == repay_60m_psrc.index.names)
    assert len(repay_60m_vals) == len(repay_60m_psrc)

    return acc_info_vals, repay_60m_vals


# %%
def test_agg_from_dfs():
    acc_info_vals, repay_60m_vals = test_flat_record()
    src = {
        "acc_info": acc_info_vals,
        "repay_60m": repay_60m_vals,
    }

    part_confs = pd.DataFrame([
        ["repay_60m"            , None                      , 1 , "rid,certno,accid,NONE"],
        ["acc_info"             , None                      , 1 , "rid,certno,accid"],
        ["repay_60m_agg"        , "repay_60m"               , 2 , "rid,certno,accid"],
        ["repay_60m_agg_agg"    , "acc_info,repay_60m_agg"  , 1 , "rid,certno"],
    ], columns=["part", "from_", "level", "prikey"])

    agg_confs = pd.DataFrame([
        ["repay_60m_agg"    , "acc_le1_mcnt"    , "acc_repay_status >= 1"  , "count(_)"],
        ["repay_60m_agg"    , "acc_le1_asum"    , "acc_repay_status >= 1"  , "sum(PD01EJ01)"],
        ["repay_60m_agg_agg", "acc_ovd_cnt"     , "acc_le1_mcnt > 0"       , "count(_)"],
        ["repay_60m_agg_agg", "acc_ovd_asum"    , None      , "sum(acc_le1_asum) * acc_exchange_rate"],
    ], columns=["part", "key", "cond", "agg"])

    trans_confs = pd.DataFrame([
        ["acc_info"     , "acc_cat"             , None              , "map(PD01AD01, cdr_cat)"],
        ["acc_info"     , "acc_exchange_rate"   , "acc_cat != 99"   , "map(PD01AD04, exchange_rate)"],
        ["repay_60m"    , "acc_repay_status"    , None              , "map(PD01ED01, repay_status)"],
    ], columns=["part", "key", "cond", "trans"])

    rets = agg_from_dfs(src, part_confs, agg_confs, trans_confs, env=MAPPER)

    assert len(rets) == part_confs["from_"].notna().sum()
