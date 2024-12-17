#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: confflat.py
#   Author: xyy15926
#   Created: 2024-12-17 09:00:28
#   Updated: 2024-12-17 10:46:52
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple

# %%
FLAT_WEIXIN = {
    "part": "flat_weixin",
    "desc": "基本信息",
    "steps": None,
    "prikey": ["rid", "certno", "tid"],
    "level": 1,
    "fields": [
        # Common columns.
        ("rid"                  , None  , "VARCHAR(255)"    , "报告编号"),
        ("certno"               , None  , "VARCHAR(255)"    , "身份证号"),
        # Report columns.
        ("tid"                  , None  , "VARCHAR(255)"    , "交易单号"),
        ("transection_time"     , None  , "DATETIME"        , "交易时间"),
        ("transection_type"     , None  , "VARCHAR(255)"    , "交易类型"),
        ("cash_inout"           , None  , "VARCHAR(255)"    , "收/支/其他"),
        ("transection_method"   , None  , "VARCHAR(255)"    , "交易方式"),
        ("transection_amount"   , None  , "FLOAT"           , "金额(元)"),
        ("counter_party"        , None  , "VARCHAR(255)"    , "交易对方"),
        ("retailer_id"          , None  , "VARCHAR(255)"    , "商户单号"),
    ]
}


# %%
FLAT_ALIPAY = {
    "part": "flat_alipay",
    "desc": "基本信息",
    "steps": None,
    "prikey": ["rid", "certno", "tid"],
    "level": 1,
    "fields": [
        # Common columns.
        ("rid"                  , None  , "VARCHAR(255)"    , "报告编号"),
        ("certno"               , None  , "VARCHAR(255)"    , "身份证号"),
        # Report columns.
        ("cash_inout"           , None  , "VARCHAR(255)"    , "收/支"),
        ("counter_party"        , None  , "VARCHAR(255)"    , "交易对方"),
        ("commodity_desc"       , None  , "VARCHAR(255)"    , "商品说明"),
        ("transection_method"   , None  , "VARCHAR(255)"    , "收/付款方式"),
        ("transection_amount"   , None  , "FLOAT"           , "金额"),
        ("tid"                  , None  , "VARCHAR(255)"    , "交易订单号"),
        ("retailer_id"          , None  , "VARCHAR(255)"    , "商家订单号"),
        ("transection_time"     , None  , "DATETIME"        , "交易时间"),
    ]
}


# %%
FLAT_CASHFLOW = {
    FLAT_WEIXIN["part"]: FLAT_WEIXIN,
    FLAT_ALIPAY["part"]: FLAT_ALIPAY,
}


def df_flat_confs():
    import pandas as pd

    parts = []
    fields = []
    for val in FLAT_CASHFLOW.values():
        part_one = {
            "part": val["part"],
            "level": val["level"],
            "prikey": val["prikey"],
            "steps": val["steps"],
            "desc": val["desc"],
        }
        parts.append(part_one)
        fields.extend([[val["part"], *ele] for ele in val["fields"]])
    parts = pd.DataFrame(parts)
    fields = pd.DataFrame.from_records(
        fields, columns=["part", "key", "step", "dtype", "desc"])

    return parts, fields
