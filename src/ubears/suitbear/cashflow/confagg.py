#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: confagg.py
#   Author: xyy15926
#   Created: 2024-12-17 09:02:19
#   Updated: 2024-12-17 15:45:05
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple

import logging
from ubears.suitbear.dirt.crosconf import cross_aggs_and_filters
from ubears.suitbear.cashflow.conftrans import (
    cond_weixin_ttype,
    cond_weixin_tmethod,
    cond_alipay_ttype,
)

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def cond_transection_amount(
    field: str = "transection_amount",
    key: str = "amt",
    desc: str = "交易金额",
    edges: list | tuple = None
) -> list:
    edges = ([10, 50, 200, 500, 1000, 2000, 5000, 10000]
             if edges is None else edges)
    lower, upper = edges[0], edges[-1]
    reprs = [(f"{key}_le{lower}",
              f"{field} < {lower}",
              f"{desc}小于{lower}")]
    reprs += [(f"{key}_{fr}to{to}",
               f"({field} >= {fr}) & ({field} < {to})",
               f"{desc}介于{fr}至{to}")
              for fr,to in zip(edges[:-1], edges[1:])]
    reprs += [(f"{key}_ge{upper}",
               f"{field} >= {upper}",
               f"{desc}大于{upper}")]
    reprs += [(None, None, None)]
    return reprs


def cond_timetag(field: str = "thour"):
    reprs = [("earlymor", f"{field} <= 6", "凌晨"),
             ("mor", f"({field} > 6) & ({field} <= 11)", "上午"),
             ("noon", f"({field} > 11) & ({field} <= 14)", "中午"),
             ("afternoon", f"({field} > 14) & ({field} <= 18)", "下午"),
             ("eve", f"({field} > 18) & ({field} <= 24)", "晚间"),
             (None, None, None)]
    return reprs


def cond_daytag_busi(field: str = "transection_time"):
    reprs = [("busi", f"is_busiday({field})", "工作日"),
             ("nbusi", f"not_busiday({field})", "节假日"),
             (None, None, None)]
    return reprs


# %%
AGG_WEIXIN = {
    "part": "agg_weixin",
    "desc": "微信流水",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["trans_weixin"],
    "joinkey": None,
    "key_fmt": "{cond}_{agg}",
    "cond": {
        "amt_threshs": cond_transection_amount("transection_amount", "amt", "交易金额"),
        "timetag": cond_timetag("thour"),
        "daytag": cond_daytag_busi("transection_time"),
        "ttype": cond_weixin_ttype("ttype"),
        "tmethod": cond_weixin_tmethod("tmethod"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "笔数"),
        "amt_sum": ("amt_sum", "sum(transection_amount)", "金额之和"),
        "amt_max": ("amt_sum", "max(transection_amount)", "最大金额"),
        "amt_avg": ("amt_sum", "avg(transection_amount)", "平均金额"),
    },
    "cros": [
        (["cnt"],
         ["amt_threshs"]),
        (["cnt", "amt_sum", "amt_max", "amt_avg"],
         ["timetag", "daytag"]),
        (["cnt", "amt_sum", "amt_max", "amt_avg"],
         ["ttype"]),
        (["cnt", "amt_sum", "amt_max", "amt_avg"],
         ["tmethod"]),
    ]
}


# %%
AGG_ALIPAY = {
    "part": "agg_alipay",
    "desc": "支付宝流水",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["trans_alipay"],
    "joinkey": None,
    "key_fmt": "{cond}_{agg}",
    "cond": {
        "amt_threshs": cond_transection_amount("transection_amount", "key", "交易金额"),
        "timetag": cond_timetag("thour"),
        "daytag": cond_daytag_busi("transection_time"),
        "ttype": cond_alipay_ttype("ttype"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "笔数"),
        "amt_sum": ("amt_sum", "sum(transection_amount)", "金额之和"),
        "amt_max": ("amt_sum", "max(transection_amount)", "最大金额"),
        "amt_avg": ("amt_sum", "avg(transection_amount)", "平均金额"),
    },
    "cros": [
        (["cnt"],
         ["amt_threshs"]),
        (["cnt", "amt_sum", "amt_max", "amt_avg"],
         ["timetag", "daytag"]),
        (["cnt", "amt_sum", "amt_max", "amt_avg"],
         ["ttype"]),
    ]
}


# %%
AGG_CONF = {
    AGG_WEIXIN["part"]: AGG_WEIXIN,
    AGG_ALIPAY["part"]: AGG_ALIPAY,
}


def df_agg_confs(confs: dict = None):
    import pandas as pd

    if confs is None:
        confs = AGG_CONF

    pconfs = []
    aconfs = {}
    for pname, pconf in confs.items():
        pname = pconf["part"]
        pconfs.append((pconf["part"],
                       pconf["level"],
                       pconf["prikey"],
                       pconf["from_"],
                       pconf.get("joinkey")))
        aconf = cross_aggs_and_filters(
            pconf["cros"], pconf["agg"], pconf["cond"], pconf["key_fmt"])
        aconfs[pname] = pd.DataFrame.from_records(
            aconf, columns=["key", "cond", "agg", "cmt"])

    # Concat the confs.
    pconfs = pd.DataFrame.from_records(
        pconfs, columns=["part", "level", "prikey", "from_", "joinkey"])
    aconfs = pd.concat(aconfs.values(), keys=aconfs.keys()).droplevel(level=1)
    aconfs.index.set_names("part", inplace=True)
    aconfs = aconfs.reset_index()

    return pconfs, aconfs
