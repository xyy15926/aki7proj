#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: conftrans.py
#   Author: xyy15926
#   Created: 2024-08-25 15:28:52
#   Updated: 2025-02-25 11:59:05
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple

import logging
import re

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
COUNTER_PARTY_KEYWORDS = {
    "travel": {
        "cat": "travel",
        "desc": "航旅机酒",
        "keywords": [
            r".*航空.*",
            r".*中国铁路网络有限公司.*",
            r".*12306.*",
            r".*酒店.*",
        ],
    },
    "drive": {
        "cat": "drive",
        "desc": "自驾出行",
        "keywords": [
            r".*代驾.*",
            r".*高速.*",
            r".*ETC.*",
            r".*石油.*",
            r".*石化.*",
            r".*加油.*",
            r".*停车.*",
        ],
    },
    "pubport": {
        "cat": "pubport",
        "desc": "公共交通",
        "keywords": [
            r".*地铁.*",
            r".*公交.*",
        ]
    },
    "taxi": {
        "cat": "taxi",
        "desc": "打车出行",
        "keywords": [
            r".*滴滴.*",
            r".*T3出行.*",
            r".*高德出行.*",
            r".*曹操出行.*",
        ],
    },
    "household": {
        "cat": "household",
        "desc": "家居消费",
        "keywords": [
            r".*燃气.*",
            r".*供电.*",
            r".*供水.*",
        ],
    },
    "finan": {
        "cat": "finan",
        "desc": "金融理财",
        "keywords": [
            r".*保险.*公司.*",
            r".*购买理财通.*",
            r".*理财通赎回.*",
        ]
    },
    "uncated": {
        "cat": "uncated",
        "desc": "未分类",
        "keywords": [".*", ]
    }
}


def compile_master_re() -> re.Pattern:
    ptns = []
    for cat, cont in COUNTER_PARTY_KEYWORDS.items():
        cat = cont["cat"]
        part_ptn = "|".join(f"(?:{ptn})" for ptn in cont["keywords"])
        ptns.append((cat, part_ptn))
    master_re = "|".join([f"(?P<{cat}>{ptn})" for cat, ptn in ptns])
    return re.compile(master_re)


COUNTER_PARTY_RE = compile_master_re()


def trans_counter_party_cats(val: str) -> str:
    mt = COUNTER_PARTY_RE.match(val)
    return mt.lastgroup


def cond_counter_part_cats(field: str = "cp_cat") -> list:
    reprs = []
    for cat, cont in COUNTER_PARTY_KEYWORDS.items():
        cat = cont["cat"]
        reprs.append((cat, f'{field} == "{cat}"', cont["desc"]))
    reprs = [(None, None, None)]
    return reprs


# %%
def trans_amount_flag(val: float) -> bool:
    RISK_TRANSECTION_AMOUNT = [
        888, 888.88, 8888, 8888.88, 666, 666.66, 6666, 6666.66, 520, 1314
    ]
    return int(val in RISK_TRANSECTION_AMOUNT)


def cond_mamt_flag(field: str = "mamt_flag") -> list:
    reprs = [("mful", f"{field} == 1", "特殊金额"),
             (None, None, None)]
    return reprs


# %%
WEIXIN_TRANSECTION_TYPE = [
    ("tf_in"         , '(transection_type == "转账") & (cash_inout == "收入")', "转入"),
    ("tf_out"        , '(transection_type == "转账") & (cash_inout == "支出")', "转出"),
    ("qrc_in"        , 'transection_type == "二维码收款"', "扫码收款"),
    ("qrc_out"       , 'transection_type == "扫二维码付款"', "扫码付款"),
    ("hb_in"         , 'isin("微信红包", transection_type) & (cash_inout == "收入")', "收红包"),
    ("hb_out"        , 'isin("微信红包", transection_type) & (cash_inout == "支出")', "发红包"),
    ("group_in"      , '(transection_type == "群收款") & (cash_inout == "收入")', "群收款收入"),
    ("group_out"     , '(transection_type == "群收款") & (cash_inout == "支出")', "群收款支出"),
    ("comsu"         , 'transection_type == "商户消费"', "商户消费"),
    ("outof"         , 'transection_type == "其他"', "其他"),
    ("credit_card"   , 'transection_type == "信用卡还款"', "信用卡还款"),
    ("change_out"    , 'transection_type == "零钱提现"', "零钱提现"),
]


def cond_weixin_ttype(field: str = "ttype"):
    reprs = [(tt, f'{field} == "{tt}"', ds)
             for tt, ts, ds in WEIXIN_TRANSECTION_TYPE]
    reprs += [(None, None, None)]
    return reprs


# %%
WEIXIN_TRANSECTION_METHOD = [
    ("credit_card"   , 'isin("信用卡", transection_method)', "信用卡"),
    ("debit_card"    , 'isin("储蓄卡", transection_method)', "借记卡"),
    ("change"        , '(transection_method == "零钱") | (transection_method == "零钱通")', "零钱"),
    ("biz_acc"       , 'transection_method == "经营账户"', "经营账户"),
]


def cond_weixin_tmethod(field: str = "fmethod"):
    reprs = [(tt, f'{field} == "{tt}"', ds)
             for tt, ts, ds in WEIXIN_TRANSECTION_METHOD]
    reprs += [(None, None, None)]
    return reprs


# %%
ALIPAY_TRANSECTION_TYPE = {
    ("comsu"    , 'retailer_id != ""', "消费"),
    ("tf_in"    , '(retailer_id == "") & (cash_inout == "收入")', "转入"),
    ("tf_out"   , '(retailer_id == "") & (cash_inout == "支出")', "转出"),
    ("qrc_out"  , 'isin("付款", commodity_desc) & (cash_inout == "支出")', "扫码付款"),
    ("qrc_in"   , 'isin("收款", commodity_desc) & (cash_inout == "收入")', "扫码收款"),
    ("outof"    , 'transection_method == "不计收支"', "不记收支"),
}


def cond_alipay_ttype(field: str = "ttype"):
    reprs = [(tt, f'{field} == "{tt}"', ds)
             for tt, ts, ds in ALIPAY_TRANSECTION_TYPE]
    reprs += [(None, None, None)]
    return reprs


# %%
TRANS_WEIXIN = {
    "part": "trans_weixin",
    "desc": "微信流水",
    "level": 1,
    "prikey": ["rid", "certno", "tid"],
    "from_": ["flat_weixin"],
    "joinkey": None,
    "trans": [
        ("cp_cat", "map(counter_party, trans_counter_party_cats)", None, "交易对手类型"),
        ("mamt_flag", "map(transection_amount, trans_amount_flag)", None, "特殊交易金额"),
        ("thour", "get_hour(transection_time)", None, "交易时"),
        *[("ttype", tt, ts, "交易类型") for tt, ts, ds in WEIXIN_TRANSECTION_TYPE],
        *[("tmethod", tt, ts, "交易方式") for tt, ts, ds in WEIXIN_TRANSECTION_METHOD],
    ]
}

TRANS_ALIPAY = {
    "part": "trans_alipay",
    "desc": "支付宝流水",
    "level": 1,
    "prikey": ["rid", "certno", "tid"],
    "from_": ["flat_alipay"],
    "joinkey": None,
    "trans": [
        ("cp_cat", "map(counter_party, trans_counter_party_cats)", None, "交易对手类型"),
        ("mamt_flag", "map(transection_amount, trans_amount_flag)", None, "特殊交易金额"),
        ("thour", "get_hour(transection_time)", None, "交易时"),
        *[("ttype", tt, ts, "交易类型") for tt, ts, ds in ALIPAY_TRANSECTION_TYPE],
    ]
}

TRANS_ENV = {
    "trans_counter_party_cats": trans_counter_party_cats,
    "trans_amount_flag": trans_amount_flag,
}

TRANS_CONF = {
    TRANS_WEIXIN["part"]: TRANS_WEIXIN,
    TRANS_ALIPAY["part"]: TRANS_ALIPAY,
}


def df_trans_confs():
    import pandas as pd

    pconfs = []
    tconfs = []
    for part_name, pconf in TRANS_CONF.items():
        part_name = pconf["part"]
        pconfs.append((pconf["part"],
                       pconf["desc"],
                       pconf["level"],
                       pconf["prikey"],
                       pconf["from_"],
                       pconf["joinkey"]))
        rules = [(part_name, key, cond, trans, desc)
                 for key, trans, cond, desc in pconf["trans"]]
        tconfs.extend(rules)

    pconfs = pd.DataFrame.from_records(
        pconfs, columns=["part", "desc", "level", "prikey",
                         "from_", "joinkey"])
    tconfs = pd.DataFrame.from_records(
        tconfs, columns=["part", "key", "cond", "trans", "desc"])

    return pconfs, tconfs
