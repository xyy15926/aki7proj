#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: confagg.py
#   Author: xyy15926
#   Created: 2024-09-23 12:10:17
#   Updated: 2024-11-20 18:27:43
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar
import logging
from itertools import product

from suitbear.dirt.crosconf import cross_aggs_and_filters
from suitbear.autofin.conftrans import (
    acc_status,
    appr_res_reprs,
    appr_code_reprs,
    biztype_cats_reprs,
)


# %%
def last_mois(field: str, desc: str = ""):
    mois = [1, 2, 3, 6, 9, 12, 18, 24, 36, 48]
    reprs = [(f"last_{moi}m",
              f"({field} >= -{moi}) & ({field} <= 0)",
              f"近{moi}月{desc}") for moi in mois]
    reprs += [(None, None, None)]
    return reprs


def last_dois(field: str, desc: str = ""):
    dois = [1, 3, 7, 15, 30, 60, 90, 180, 365, 730]
    reprs = [(f"last_{doi}d",
              f"({field} >= -{doi}) & ({field} <= 0)",
              f"近{doi}日{desc}")
             for doi in dois]
    reprs += [(None, None, None)]
    return reprs


# %%
def amt_bins_1(field: str = "loan_term",
               desc: str = "期限",
               edges: list | tuple = None):
    edges = [12, 24, 36, 48] if edges is None else edges
    lower, upper = edges[0], edges[-1]
    reprs = [(f"{field}_le{lower}",
              f"{field} < {lower}",
              f"{desc}小于{lower}")]
    reprs += [(f"{field}_{fr}to{to}",
               f"({field} >= {fr}) & ({field} < {to})",
               f"{desc}介于{fr}至{to}")
              for fr,to in zip(edges[:-1], edges[1:])]
    reprs += [(f"{field}_ge{upper}",
               f"{field} >= {upper}",
               f"{desc}大于{upper}")]
    reprs += [(None, None, None)]
    return reprs


def amt_bins_w(field: str = "loan_amt",
               desc: str = "融资额",
               edges: list | tuple = None):
    edges = [5, 10, 20, 30, 50] if edges is None else edges
    lower, upper = edges[0], edges[-1]
    reprs = [(f"{field}_le{lower}w",
              f"{field} < {lower}",
              f"{desc}小于{lower}万")]
    reprs += [(f"{field}_{fr}to{to}w",
               f"({field} >= {fr}) & ({field} < {to})",
               f"{desc}介于{fr}至{to}万")
              for fr,to in zip(edges[:-1], edges[1:])]
    reprs += [(f"{field}_ge{upper}w",
               f"{field} >= {upper}",
               f"{desc}大于{upper}万")]
    reprs += [(None, None, None)]
    return reprs


def amt_threshs_1(key: str, field: str, desc: str, upper: int = None):
    amt_threshs = [1, 2, 3, 5, 7, 10]
    amt_threshs = [5, 7]
    if field is None:
        field = key
    if upper is not None:
        amt_threshs = [i for i in amt_threshs if i < upper] + [upper, ]
    reprs = [(f"{key}_ge{amt}", f"{field} >= {amt}", f"{desc}大于等于{amt}")
             for amt in amt_threshs]
    reprs += [(None, None, None),]
    return reprs


# %%
def none_of_all():
    reprs = [(None, None, None), ]
    return reprs


# %%
def _basic_upper_aggs(lower_conf: dict, desc: str = "账户"):
    upper_aggs = {}
    lower_agg_conf = cross_aggs_and_filters(
        lower_conf["cros"],
        lower_conf["agg"],
        lower_conf["cond"],
        lower_conf["key_fmt"]
    )

    lvup_fmt = {
        "max": ("{}_max", "max({})", "{}" + desc + "最大值"),
        "min": ("{}_min", "min({})", "{}" + desc + "最小值"),
        "sum": ("{}_sum", "sum({})", "{}" + desc + "之和"),
    }
    lvup_refs = {
        "cnt": ["max", "sum"],
        "max": ["max"],
        "min": ["min"],
        "sum": ["max", "sum"],
        "por": ["max"],
        "lmt": ["max", "sum"],
        "usd": ["max", "sum"],
        "ots": ["max", "sum"],
        "ovdd": ["max", "sum"],
        "ovdt": ["max", "sum"],
        "ovdo": ["max", "sum"],
        "ovdp": ["max", "sum"],
    }
    for key, _cond, _agg, desc in lower_agg_conf:
        agg_type = key[-3:]
        for lvup_type in lvup_refs.get(agg_type, []):
            lvup_key, lvup_agg, lvup_desc = lvup_fmt[lvup_type]
            upper_aggs[lvup_key.format(key)] = (
                lvup_key.format(key),
                lvup_agg.format(key),
                lvup_desc.format(desc)
            )
    return upper_aggs


# %%
LOAN_REPAYMENT = {
    "part": "loan_repayment",
    "desc": "账户按月还款情况",
    "prikey": ["certno", "order_no"],
    "level": 1,
    "from_": ["loan_repayment_monthly"],
    "key_fmt": "repay_{cond}_{agg}",
    "cond": {
        "mois": last_mois("duepay_moi"),
        "ovdd": amt_threshs_1("ovd_days", "ovd_days", "逾期天数"),
        "None": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "期数"),
        # 历史还款，未包含：最大连续逾期期数、逾期期数和、最大逾期金额之类
        "ovdd_max": ("ovdd_max", "max(ovd_days)", "最大逾期天数"),
        "fpd_ovdd": ("fpd_ovdd", "getn(sortby(ovd_days, mob), 0)", "首期逾期天数"),
        # 当前状态
        "cur_ovdd": ("cur_ovdd_cnt", "max(ovd_days[isnull(repay_date)])", "当前逾期天数"),
        "cur_ovdt": ("cur_ovdt_cnt", "count(ovd_days[isnull(repay_date)])", "当前逾期期数"),
        "cur_ovdo": ("cur_ovdo_sum", "sum(duepay_amt[isnull(repay_date) & (ovd_days > 0)])", "当前逾期金额"),
        "cur_ovdp": ("cur_ovdp_sum", "sum(duepay_pri[isnull(repay_date) & (ovd_days > 0)])", "当前逾期本金"),
        "remo_sum": ("remo_sum", "sum(duepay_amt[isnull(repay_date)])", "剩余未还金额"),
        "remp_sum": ("remp_sum", "sum(duepay_pri[isnull(repay_date)])", "剩余未还本金"),
    },
    "cros": [
        (["cnt",],
         ["mois", "ovdd"]),
        (["ovdd_max"],
         ["mois",]),
        (["fpd_ovdd"],
         ["None",]),
        (["cur_ovdd", "cur_ovdt", "cur_ovdo", "cur_ovdp",
          "remo_sum", "remp_sum"],
         ["None"])
    ],
}


# %%
AGG_LOAN_REPAYMENT = {
    "part": "agg_loan_repayment",
    "desc": "客户还款情况",
    "prikey": ["certno",],
    "level": 0,
    "from_": ["loan_acc_info", "loan_repayment"],
    "joinkey": [["certno", "order_no"],
                ["certno", "order_no"]],
    "key_fmt": "{cond}_{agg}",
    "cond": {
        "loan_pri": amt_bins_w("loan_pri", "融资额"),
        "loan_term": amt_bins_1("loan_term", "期限"),
        "loan_ppor": amt_bins_1("loan_ppor", "融资比例"),
    },
    "agg": _basic_upper_aggs(LOAN_REPAYMENT),
    "cros": [
        (list(_basic_upper_aggs(LOAN_REPAYMENT).keys()),
         ["loan_pri"]),
        (list(_basic_upper_aggs(LOAN_REPAYMENT).keys()),
         ["loan_term"]),
        (list(_basic_upper_aggs(LOAN_REPAYMENT).keys()),
         ["loan_ppor"]),
    ],
}


# %%
AGG_LOAN_ACC_INFO = {
    "part": "agg_loan_acc_info",
    "desc": "放款情况",
    "prikey": ["certno",],
    "level": 0,
    "from_": ["loan_acc_info"],
    "key_fmt": "acc_{cond}_{agg}",
    "cond": {
        "status": acc_status("acc_status"),
        "None": [(None, None, None)],
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "账户数"),
        "last_loans_mcnt": ("last_loans_dcnt", "min(-acc_start_moi)", "上次放款距今月"),
        "last_loane_mcnt": ("last_loane_dcnt", "min(-acc_end_moi)", "上次关闭距今月"),
    },
    "cros": [
        (["cnt",],
         ["status"]),
        (["last_loans_mcnt", "last_loane_mcnt"],
         ["None"]),
    ],
}


# %%
AGG_AUTOFIN_PRETRIAL = {
    "part": "agg_autofin_pretrial",
    "desc": "预审情况",
    "prikey": ["certno"],
    "level": 0,
    "from_": ["autofin_pretrial"],
    "key_fmt": "pretrial_{cond}_{agg}",
    "cond": {
        "biztype_cats": biztype_cats_reprs("biztype"),
        "approval_result": appr_res_reprs("appr_res"),
        "infocode_cats": appr_code_reprs("infocode_cats"),
        "dois": last_dois("apply_doi"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "预审数"),
        "last_apply_dcnt": ("last_apply_dcnt",
                            "min(-apply_doi)",
                            "上次预审距今日"),
    },
    "cros": [
        (["cnt",],
         ["biztype_cats", "approval_result", "infocode_cats", "dois"]),
        (["last_apply_dcnt"],
         ["biztype_cats", "approval_result"]),
    ]
}


# %%
AGG_AUTOFIN_SECTRIAL = {
    "part": "agg_autofin_sectrial",
    "desc": "资审申请情况",
    "prikey": ["certno"],
    "level": 0,
    "from_": ["autofin_sectrial"],
    "key_fmt": "sectrial_{cond}_{agg}",
    "cond": {
        "biztype_cats": biztype_cats_reprs("biztype"),
        "approval_result": appr_res_reprs("appr_res"),
        "infocode_cats": appr_code_reprs("infocode_cats"),
        "dois": last_dois("apply_doi"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "资审数"),
        "last_apply_dcnt": ("last_apply_dcnt",
                            "min(-apply_doi)",
                            "上次资审距今日"),
    },
    "cros": [
        (["cnt",],
         ["biztype_cats", "approval_result", "infocode_cats", "dois"]),
        (["last_apply_dcnt"],
         ["biztype_cats", "approval_result"]),
    ]
}


# %%
MASS_AUTOFIN_PRETRIAL = {
    "part": "mass_autofin_pretrial",
    "desc": "预审情况汇总",
    "prikey": ["channel_code"],
    "level": 0,
    "from_": ["autofin_pretrial"],
    "key_fmt": "mpretrial_{cond}_{agg}",
    "cond": {
        "biztype_cats": biztype_cats_reprs("biztype"),
        "approval_result": appr_res_reprs("appr_res"),
        "infocode_cats": appr_code_reprs("infocode_cats"),
        "dois": last_dois("apply_doi"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        (["cnt"],
         ["approval_result", "infocode_cats", "dois"]),
    ]
}


MASS_AUTOFIN_SECTRIAL = {
    "part": "mass_autofin_sectrial",
    "desc": "预审情况汇总",
    "prikey": ["channel_code"],
    "level": 0,
    "from_": ["autofin_sectrial"],
    "key_fmt": "mpretrial_{cond}_{agg}",
    "cond": {
        "biztype_cats": biztype_cats_reprs("biztype"),
        "approval_result": appr_res_reprs("appr_res"),
        "infocode_cats": appr_code_reprs("infocode_cats"),
        "dois": last_dois("apply_doi"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        (["cnt"],
         ["approval_result", "infocode_cats", "dois"]),
    ]
}


# %%
GROUP_AUTOFIN_PRETRIAL_TMPL = {
    "part": "group_autofin_pretrial_{}",
    "desc": "预审情况汇总",
    "prikey": None,
    "level": 0,
    "from_": ["autofin_pretrial"],
    "key_fmt": "gpretrial_{cond}_{agg}",
    "cond": {
        "biztype_cats": biztype_cats_reprs("biztype"),
        "approval_result": appr_res_reprs("appr_res"),
        "infocode_cats": appr_code_reprs("infocode_cats"),
        "dois": last_dois("apply_doi"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        (["cnt"],
         ["approval_result", "infocode_cats",]),
    ]
}


GROUP_AUTOFIN_SECTRIAL_TMPL = {
    "part": "group_autofin_sectrial_{}",
    "desc": "预审情况汇总",
    "prikey": None,
    "level": 0,
    "from_": ["autofin_sectrial"],
    "key_fmt": "gsectrial_{cond}_{agg}",
    "cond": {
        "biztype_cats": biztype_cats_reprs("biztype"),
        "approval_result": appr_res_reprs("appr_res"),
        "infocode_cats": appr_code_reprs("infocode_cats"),
        "dois": last_dois("apply_doi"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        (["cnt"],
         ["approval_result", "infocode_cats",]),
    ]
}


def group_with_prikey(tmpl: dict):
    prikey_M = {
        "p1": ["channel_code", "apply_date", "biztype"],
    }
    confs = {}
    for kname, prikey in prikey_M.items():
        conf = tmpl.copy()
        conf["part"] = conf["part"].format(kname)
        conf["prikey"] = prikey
        confs[conf["part"]] = conf

    return confs


# %%
PERSONAL_CONF = {
    LOAN_REPAYMENT["part"]          : LOAN_REPAYMENT,
    AGG_LOAN_REPAYMENT["part"]      : AGG_LOAN_REPAYMENT,
    AGG_LOAN_ACC_INFO["part"]       : AGG_LOAN_ACC_INFO,
    AGG_AUTOFIN_PRETRIAL["part"]    : AGG_AUTOFIN_PRETRIAL,
    AGG_AUTOFIN_SECTRIAL["part"]    : AGG_AUTOFIN_SECTRIAL,
}

MASS_CONF = {
    MASS_AUTOFIN_PRETRIAL["part"]   : MASS_AUTOFIN_PRETRIAL,
    MASS_AUTOFIN_SECTRIAL["part"]   : MASS_AUTOFIN_SECTRIAL,
}


# %%
def df_agg_confs(confs: dict = None):
    import pandas as pd
    if confs is None:
        confs = {**PERSONAL_CONF, **MASS_CONF}

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
