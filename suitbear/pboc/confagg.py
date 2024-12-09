#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: confagg.py
#   Author: xyy15926
#   Created: 2024-09-10 10:52:24
#   Updated: 2024-12-08 17:11:59
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import TypeVar
from itertools import product
from suitbear.pboc.conftrans import (
    repay_status,
    repay_status_spec,
    special_trans,
    special_accd,
    acc_cat,
    biz_cat,
    org_cat,
    acc_status,
    lvl5_status,
    trans_status,
    guar_type,
    credit_cat,
    postfee_acc_cat,
    postfee_acc_status,
    housing_fund_status,
    inq_rec_reason_cat,
    marital_status,
    res_status,
    comp_char,
    comp_indust,
    comp_prof,
    comp_pos,
    comp_prof_title,
)


# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def cproduct_aggs_and_filters(
    aggs: list | dict,
    filters: list,
    key_fmt: str = "{cond}_{agg}",
) -> list[tuple]:
    """Generate Cartesian Product for aggregations and filters.

    Each aggregation and all filters will be looped to get the Cartesian
    Product as the configuration.
    1. Cartesian Product of condition-groups will be generated with the
      filters in `filters` without aggregations.
    2. Aggregations and the condition-groups will be grouped together later
      for each aggregation to get configuration of how to aggregate.

    Params:
    -----------------------
    aggs: List of 3-Tuples[NAME, AGG-EXPR, COMMENT] describing the
      aggregations.
    filters: List of 3-Tuples[NAME, COND-EXPR, COMMENT] describing the
      the filters.
    key_fmt: Formation string for constructing the key's names.

    Return:
    -----------------------
    List of 4-Tuple[KEY, COND-EXPR, AGG-EXPR, CMT]
    """
    conds = []
    # Outer Cartesian Product of aggregations and condition-groups.
    for agg_varn, agg_fn, agg_cmt in aggs:
        # Inner Cartesian Product of conditions.
        for cond_grp in product(*filters):
            # Zip to gather the list of NAMEs, CONDITIONs and COMMENTs.
            # And filter `NONE` in NAMEs, CONDITIONs and COMMENTs.
            cond_varn, cond_cond, cond_cmt = [
                list(filter(lambda x: x is not None, ele))
                for ele in zip(*cond_grp)
            ]
            cond_varn_str = "_".join(cond_varn)
            cond_cond_str = " & ".join([f"({ele})" for ele in cond_cond])
            cond_cmt_str = "".join(cond_cmt)

            # `_` will be repeated in some formation string with no filter set.
            key_str = (key_fmt.format(cond=cond_varn_str, agg=agg_varn)
                       .strip("_")
                       .replace("__", "_"))
            conds.append((key_str, cond_cond_str, agg_fn,
                          cond_cmt_str + agg_cmt))
    return conds


# %%
def cross_aggs_and_filters(
    cros: tuple[list, list],
    aggs_D: dict[str, list],
    filters_D: dict[str, list | dict],
    key_fmt: str = "{cond}_{agg}",
) -> list[tuple]:
    """Cross the aggregations and filters according to `cros`.

    There may be many aggregations and filters in some area while exploring the
    data. But not all of them can be combined together with simple Cartesian
    Product or not all the branches in a filter are compatible with the
    aggregation.
    So `cros` is provided here to specify the reasonable pairs of aggregation
    and filter, and Cartesian Product will be done only on the specified pairs.

    Params:
    ----------------------
    cros: 2-Tuple[AGG-CODE-LIST, FILTER-CODE-LIST], indicating how to pair
      aggregations and filters.
      AGG-CODE-LIST: List of str indicating the aggregations.
      FILTER-CODE-LIST: List of str or tuple indicating the filters.
        tuple: [FILTER-CODE, FILTER-BRANCHES,...]
        str-tuple: <FILTER-CODE>.[FILTRE-BRANCHES,...]
        simple str: filter-code from `filters_D`
    aggs_D: Dict[AGG-CODE, aggregation description], from which to get
      the aggregation description with the code.
    filters_D: Dict[FILTER-CODE, dict or list of filter description], from
      which to get the filter description with the code.
      list: [(NAME, FILTER, COMMENT), ...]
      dict: {FILTER-BRANCH: (NAME, FILTER, COMMENT), ...}

    Return:
    ----------------------
    DataFrame with columns[key, conds, agg, cmt], which fits for
      `fxgine.agg_from_dfs`.
    """
    confs = []
    for agg_codes, filter_indcs in cros:
        # Get filter descriptions from the `filters_D`.
        filters_ = []
        for filter_indc in filter_indcs:
            # For filter indicator with format: (key, sub_key, sub_key,...)
            if isinstance(filter_indc, (tuple, list)):
                filter_code, *branches = filter_indc
            # For filter indicator with format: "key.[sub_key,sub_key]"
            else:
                filter_code, *branches = filter_indc.split(".")
                if branches:
                    branches = [ele.strip() for ele in
                                branches[0][1: -1].split(",")]

            filter_ = filters_D[filter_code]
            # Choose specified branches only.
            if branches:
                filters_.append([filter_[br] for br in branches])
            # Choose all branches from dict or list.
            elif isinstance(filter_, dict):
                filters_.append(list(filter_.values()))
            else:
                filters_.append(filter_)

        # Get aggregation descriptions from the `aggs_D`.
        aggs_ = [aggs_D[an] for an in agg_codes]

        # Generate Cartesian Product of aggregations and filters.
        conds = cproduct_aggs_and_filters(aggs_, filters_, key_fmt)
        confs.extend(conds)

    # Drop duplicated conf items.
    conf_dict = {}
    for key, cond, agg, cmt in confs:
        conf_dict.setdefault(key, (cond, agg, cmt))
    confs = [(key, *val) for key, val in conf_dict.items()]

    return confs


# %%
#TODO
def last_mois(field: str, desc: str = ""):
    mois = [1, 2, 3, 6, 9, 12, 18, 24, 36, 48]
    reprs = [(f"last_{moi}m",
              f"({field} >= -{moi}) & ({field} <= 0)",
              f"近{moi}月{desc}") for moi in mois]
    # 已上线字段可能分别使用 `all`、`his`，不好统一:(
    reprs += [("all", None, "历史"),
              ("his", None, "历史"),
              (None, None, None)]
    return reprs


def last_dois(field: str, desc: str = ""):
    dois = [1, 2, 3, 5, 10, 15, 20, 30, 60, 90, 180, 270, 365, 730, 1460]
    reprs = [(f"last_{doi}d",
              f"({field} >= -{doi}) & ({field} <= 0)",
              f"近{doi}日{desc}") for doi in dois]
    # 已上线字段可能分别使用 `all`、`his`，不好统一:(
    reprs += [("all", None, "历史"),
              ("his", None, "历史"),
              (None, None, None)]
    return reprs


def folw_mois(field: str, desc: str = "存续"):
    mois = [3, 6, 9, 12, 18, 24, 36, 48]
    reprs = [(f"folw_{moi}m",
              f"({field} <= {moi}) & ({field} > 0)",
              f"未来{moi}月{desc}") for moi in mois]
    reprs += [("closed", f"{field} <= 0", "未{desc}"),
              ("open", f"{field} > 0", "仍{desc}"),
              (None, None, None)]
    return reprs


# %%
def amt_threshs_1(key: str, field: str, desc: str, upper: int = None):
    amt_threshs = [0, 1, 2, 3, 5, 7, 10]
    if field is None:
        field = key
    if upper is not None:
        amt_threshs = [i for i in amt_threshs if i < upper] + [upper, ]
    reprs = [(f"{key}_ge{amt}", f"{field} > {amt}", f"{desc}大于{amt}")
             for amt in amt_threshs]
    reprs = [(f"{key}_eq{amt}", f"{field} == {amt}", f"{desc}等于{amt}")
             for amt in amt_threshs]
    reprs += [(None, None, None),]
    return reprs


def amt_threshs_100(key: str, field: str, desc: str, upper: int = None):
    amt_threshs = [100, 200, 500, 1000]
    if field is None:
        field = key
    if upper is not None:
        amt_threshs = [i for i in amt_threshs if i < upper] + [upper, ]
    reprs = [(f"{key}_ge{amt}", f"{field} > {amt}", f"{desc}大于{amt}")
             for amt in amt_threshs]
    reprs += [(None, None, None),]
    return reprs


def amt_threshs_w(key: str, field: str, desc: str, upper: int = None):
    amt_threshs = [1, 5, 10, 20, 30, 50]
    if field is None:
        field = key
    if upper is not None:
        amt_threshs = [i for i in amt_threshs if i < upper] + [upper, ]
    reprs = [(f"{key}_ge{amt}w", f"{field} > {amt * 10000}", f"{desc}大于{amt}万")
             for amt in amt_threshs]
    reprs += [(None, None, None),]
    return reprs


def ppor_threshs(key: str, field: str, desc: str, upper: int = None):
    ppor_threshs = [20, 50, 80, 100]
    if field is None:
        field = key
    if upper is not None:
        ppor_threshs = [i for i in ppor_threshs if i < upper] + [upper, ]
    reprs = [(f"{key}_ge{rt}", f"{field} > {rt/100}", f"{desc}大于{rt/100}")
             for rt in ppor_threshs]
    reprs += [(None, None, None),]
    return reprs


# %%
def none_of_all():
    reprs = [(None, None, "所有记录中"), ]
    return reprs


# %%
def _basic_upper_aggs(lower_conf: dict, desc: str = "各账户"):
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
ACC_REPAY_60M = {
    "part": "acc_repay_60m",
    "desc": "账户60个月还款",
    "level": 1,
    "prikey": ["rid", "certno", "accid"],
    "from_": ["pboc_acc_repay_60_monthly"],
    "joinkey": None,
    "key_fmt": "acc_repay_{cond}_{agg}",
    "cond": {
        "mois": last_mois("acc_repay_moi"),
        "status": (repay_status("acc_repay_status")
                   + repay_status_spec("acc_repay_status_spec")),
        "amt_threshs": amt_threshs_100("ovd_amt", "PD01EJ01", "逾期金额"),
    },
    # "agg": Dict[agg_key, Tuple[name, agg_func, desc, List[upper_agg,...]]]
    # `upper_agg` 指定再上层可聚集操作，具体聚集操作由再上层配置决定
    "agg": {
        "cnt": ("cnt", "count(_)", "期数"),
        "status_max": ("status_max", "max(acc_repay_status)", "最大还款状态"),
        "status8_max": ("status8_max", "max(acc_repay_status8)", "最大还款状态8"),
        "status_ovd_conl_max": (
            "status_ovd_conl_max",
            "flat1_max(sortby(acc_repay_status, PD01ER03, 1) > 0)",
            "最长连续逾期期数"),
        "status_sum": ("status_sum", "sum(acc_repay_status)", "还款状态之和"),
        "ovd_max": ("ovd_max", "max(PD01EJ01)", "最大逾期（透支）金额"),
        "ovd_sum": ("ovd_sum", "sum(PD01EJ01)", "逾期（透支）金额之和"),
        "last_ovd_prd_max": (
            "last_ovd_prd_max",
            "max(-acc_repay_moi)",
            "最早逾期距今（月）"),
        "last_ovd_prd_min": (
            "last_ovd_prd_min",
            "min(-acc_repay_moi)",
            "最晚逾期距今（月）"),
    },
    # "cros": Tuple[[agg, ...], [cond,...]]
    # 各 `agg` 项分别与全部 `cond` 的笛卡尔积作笛卡尔积
    # 即：agg * product(cond1, cond2, ...)
    "cros": [
        (["cnt", "ovd_max", "ovd_sum",
          "last_ovd_prd_max", "last_ovd_prd_min"],
         ["mois", "status"]),
        (["status_max", "status_sum", "status_ovd_conl_max",
          "status8_max"],
         ["mois", ]),
    ],
}


# %%
ACC_REPAY_60M_AGG_BASIC = {
    "part": "acc_repay_60m_agg_basic",
    "desc": "报文60个月还款情况基本",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["acc_repay_60m", "pboc_acc_info"],
    "joinkey": [["rid", "certno", "accid"],
                ["rid", "certno", "accid"]],
    # 已上线字段已使用 `key_fmt`，不方便再修改:(
    "key_fmt": "{cond}{agg}",
    "cond": {
        "acc_cat": acc_cat("acc_cat", ["r2", "d1r41", "r23", "r1", "d1"]),
        "biz_cat": biz_cat("acc_biz_cat"),
        "orgs": org_cat("acc_org_cat"),
    },
    "agg": _basic_upper_aggs(ACC_REPAY_60M),
    "cros": [
        (list(_basic_upper_aggs(ACC_REPAY_60M).keys()),
         ["acc_cat"]),
        (list(_basic_upper_aggs(ACC_REPAY_60M).keys()),
         ["biz_cat"]),
        (list(_basic_upper_aggs(ACC_REPAY_60M).keys()),
         ["orgs"]),
    ],
}

ACC_REPAY_60M_AGG_ADV = {
    "part": "acc_repay_60m_agg_adv",
    "desc": "报文60个月还款情况特化",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["acc_repay_60m", "pboc_acc_info"],
    "joinkey": [["rid", "certno", "accid"],
                ["rid", "certno", "accid"]],
    # 为调整 `key_fmt` 拆分
    "key_fmt": "{cond}_{agg}",
    "cond": {
        "acc_cat": acc_cat("acc_cat", ["r2", "d1r41", "r23"]),
        "acc_status": acc_status("mixed_acc_status"),
        "acc_repay_le1_cnt": amt_threshs_1("acc_repay_le1_cnt",
                                           "acc_repay_le1_cnt",
                                           "历史逾期次数")
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "账户数"),
        "last_prd_sum": ("last_prd_sum", "sum(-acc_moi_start)", "距今月份和"),
    },
    "cros": [
        (["cnt", "last_prd_sum"],
         ["acc_cat", "acc_status", "acc_repay_le1_cnt"])
    ],
}


# %%
ACC_SPECIAL_TRANS = {
    "part": "acc_special_trans",
    "desc": "账户特殊交易",
    "level": 1,
    "prikey": ["rid", "certno", "accid"],
    "from_": ["pboc_acc_special_trans"],
    "joinkey": None,
    "key_fmt": "acc_special_trans_{cond}_{agg}",
    "cond": {
        "mois": last_mois("acc_special_trans_moi"),
        "trans": special_trans("acc_special_trans_type"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "特殊交易记录数"),
        "sum": ("sum", "sum(PD01FJ01)", "特殊交易额度和"),
        "max": ("max", "max(PD01FJ01)", "特殊交易额度最大值"),
    },
    "cros": [
        (["cnt", "sum", "max"]              , ["mois", "trans"]),
    ],
}


# %%
ACC_SPECIAL_TRANS_AGG = {
    "part": "acc_special_trans_agg",
    "desc": "报文特殊交易",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["acc_special_trans", "pboc_acc_info"],
    "joinkey": [["rid", "certno", "accid"],
                ["rid", "certno", "accid"]],
    "key_fmt": "{cond}{agg}",
    "cond": {
        "acc_cat": acc_cat("acc_cat", ["c1", "d1r41", "r23"]),
    },
    "agg": _basic_upper_aggs(ACC_SPECIAL_TRANS),
    "cros": [
        (list(_basic_upper_aggs(ACC_SPECIAL_TRANS).keys()),
         ["acc_cat"]),
    ],
}


# %%
ACC_SPECIAL_ACCD = {
    "part": "acc_special_accd",
    "desc": "账户特殊事件",
    "level": 1,
    "prikey": ["rid", "certno", "accid"],
    "from_": ["pboc_acc_special_accd"],
    "joinkey": None,
    "key_fmt": "acc_special_accd_{cond}_{agg}",
    "cond": {
        "mois": last_mois("acc_special_accd_moi"),
        "trans": special_accd("acc_special_accd_type"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "特殊事件记录数"),
    },
    "cros": [
        (["cnt", ]                          , ["mois", ]),
    ],
}


# %%
ACC_SPECIAL_ACCD_AGG = {
    "part": "acc_special_accd_agg",
    "desc": "报文特殊事件",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["acc_special_accd", "pboc_acc_info"],
    "joinkey": [["rid", "certno", "accid"],
                ["rid", "certno", "accid"]],
    "key_fmt": "{cond}{agg}",
    "cond": {
        "acc_cat": acc_cat("acc_cat", ["r2"]),
    },
    "agg": _basic_upper_aggs(ACC_SPECIAL_ACCD),
    "cros": [
        (list(_basic_upper_aggs(ACC_SPECIAL_ACCD).keys()),
         ["acc_cat"]),
    ],
}


# %%
ACC_SPECIAL_INSTS = {
    "part": "acc_special_insts",
    "desc": "账户专项分期",
    "level": 1,
    "prikey": ["rid", "certno", "accid"],
    "from_": ["pboc_acc_special_insts"],
    "joinkey": None,
    "key_fmt": "acc_special_insts_{cond}_{agg}",
    "cond": {
        "mois_start": last_mois("acc_special_insts_moi_start"),
        "mois_end": folw_mois("acc_special_insts_moi_end"),
        "None": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "大额专项分期记录数"),
        "lmt_sum": ("lmt_sum", "sum(PD01HJ01)", "大额专项分期额度和"),
        "lmt_max": ("lmt_max", "sum(PD01HJ01)", "大额专项分期额度最大值"),
        "usd_sum": ("usd_sum", "sum(PD01HJ02)", "大额专项分期已用额度和"),
        "usd_max": ("usd_max", "max(PD01HJ02)", "大额专项分期已用额度最大值"),
        "usd_sum_ppor": (
            "usd_sum_ppor",
            "sdiv(sum(PD01HJ02), sum(PD01HJ01))",
            "大额专项分期已用额度占比"
        ),
        # "monthly_repay_max": (
        #     "monthly_repay_max",
        #     "max(acc_special_insts_monthly_repay)",
        #     "大额分期月还款最大值"
        # ),
        "monthly_repay_sum": (
            "monthly_repay_sum",
            "sum(acc_special_insts_monthly_repay)",
            "大额分期月还款之和"
        ),
    },
    "cros": [
        (["cnt", "lmt_sum", "lmt_max",
          "usd_sum", "usd_max", "usd_sum_ppor",
          "monthly_repay_sum"],
         ["mois_end", ]),
    ],
}


# %%
ACC_SPECIAL_INSTS_AGG = {
    "part": "acc_special_insts_agg",
    "desc": "报文专项分期",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["acc_special_insts", "pboc_acc_info"],
    "joinkey": [["rid", "certno", "accid"],
                ["rid", "certno", "accid"]],
    "key_fmt": "{cond}{agg}",
    "agg": _basic_upper_aggs(ACC_SPECIAL_INSTS),
    "cond": {
        "acc_cat": acc_cat("acc_cat", ["r2", "r281", "r282", "r2spec"]),
    },
    "cros": [
        (list(_basic_upper_aggs(ACC_SPECIAL_INSTS).keys()),
         ["acc_cat"]),
    ],
}


# %%
LV2_AGG_CONF = {
    ACC_REPAY_60M["part"]           : ACC_REPAY_60M,
    ACC_SPECIAL_ACCD["part"]        : ACC_SPECIAL_ACCD,
    ACC_SPECIAL_TRANS["part"]       : ACC_SPECIAL_TRANS,
    ACC_SPECIAL_INSTS["part"]       : ACC_SPECIAL_INSTS,
}

LV20_AGG_CONF = {
    ACC_REPAY_60M_AGG_BASIC["part"] : ACC_REPAY_60M_AGG_BASIC,
    ACC_REPAY_60M_AGG_ADV["part"]   : ACC_REPAY_60M_AGG_ADV,
    ACC_SPECIAL_ACCD_AGG["part"]    : ACC_SPECIAL_ACCD_AGG,
    ACC_SPECIAL_INSTS_AGG["part"]   : ACC_SPECIAL_INSTS_AGG,
    ACC_SPECIAL_TRANS_AGG["part"]   : ACC_SPECIAL_TRANS_AGG,
}


# %%
PINFO_MOBILE = {
    "part": "pinfo_mobile",
    "desc": "报文手机号",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_mobile"],
    "joinkey": None,
    "key_fmt": "pinfo_mobile_{cond}_{agg}",
    "cond": {
        "mois": last_mois("mon_itvl(PB01BR01, today)"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "手机号数量"),
    },
    "cros": [
        (["cnt", ]                          , ["mois", ]),
    ]
}


# %%
PINFO_RES = {
    "part": "pinfo_res",
    "desc": "报文住址",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_address"],
    "joinkey": None,
    "key_fmt": "pinfo_res_{cond}_{agg}",
    "cond": {
        "mois": last_mois("mon_itvl(PB030R01, today)"),
        "res_status": res_status("pi_res_status"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "居住地数量"),
    },
    "cros": [
        (["cnt", ]                          , ["mois", ]),
        (["cnt", ]                          , ["res_status", ]),

    ]
}


# %%
PINFO_COMP = {
    "part": "pinfo_comp",
    "desc": "报文工作",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_company"],
    "joinkey": None,
    "key_fmt": "pinfo_comp_{cond}_{agg}",
    "cond": {
        "mois": last_mois("mon_itvl(PB040R01, today)"),
        "comp_char": comp_char("pi_comp_char"),
        "comp_indust": comp_indust("pi_comp_indust"),
        "comp_prof": comp_prof("pi_comp_prof"),
        "comp_pos": comp_pos("pi_comp_pos"),
        "comp_prof_title": comp_prof_title("pi_comp_prof_title"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "工作单位数量"),
    },
    "cros": [
        (["cnt", ]                          , ["mois", ]),
        (["cnt", ]                          , ["comp_char", ]),
        (["cnt", ]                          , ["comp_indust", ]),
        (["cnt", ]                          , ["comp_prof", ]),
        (["cnt", ]                          , ["comp_pos", ]),
        (["cnt", ]                          , ["comp_prof_title", ]),
    ]
}


# %%
ACC_INFO = {
    "part": "acc_info",
    "desc": "报文账户信息",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_acc_info"],
    "joinkey": None,
    "key_fmt": "acc_{cond}_{agg}",
    "cond": {
        "mois_start": last_mois("acc_moi_start", "开立"),
        "mois_end": folw_mois("acc_moi_end"),
        "acc_cat": acc_cat("acc_cat"),
        "acc_cat_simple": acc_cat("acc_cat", ["d1", "r4", "r1", "r2"]),
        "acc_cat_r23": acc_cat("acc_cat", ["r2", "r2cny", "r23", "r281", "r282"]),
        "acc_cat_r2": acc_cat("acc_cat", ["r2", "r2cny", "r281", "r282"]),
        "acc_cat_dr41": acc_cat("acc_cat", ["d1", "r4", "r1", "d1r4", "d1r41"]),
        "acc_cat_dr412": acc_cat("acc_cat", ["d1", "r4", "r1", "d1r4", "d1r41",
                                             "r2", "r2cny", "r281", "r282"]),
        "acc_cat_nc": acc_cat("acc_cat", ["d1", "r4", "r1", "d1r4", "d1r41",
                                          "r2", "r2cny", "r23", "r281", "r282"]),
        "acc_cat_c": acc_cat("acc_cat", ["c1"]),
        # 账户模块中，`org_bank` 上线字段中意为 “商业银行”
        # 但查询模块中，`org_bank` 上线字段中意为 “银行机构”
        "orgs": (org_cat("acc_org_cat")
                 + [("org_bank", "acc_org_cat == 11", "商业银行")]),
        "biz_cat": biz_cat("acc_biz_cat"),
        "mixed_acc_status": acc_status("mixed_acc_status"),
        "mixed_lvl5_status": lvl5_status("mixed_lvl5_status"),
        "trans_status": trans_status("acc_trans_status"),
        "guar_type": guar_type("acc_guar_type"),
        "usd_ppor": ppor_threshs("monthly_usd_ppor", "monthly_usd_ppor",
                                 "月度额度使用率"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "账户数"),
        "lmt_sum": ("lmt_sum", "sum(acc_lmt)", "借款、授信额度之和"),
        "lmt_max": ("lmt_max", "max(acc_lmt)", "借款、授信额度最大值"),
        "lmt_min": ("lmt_min", "min(acc_lmt)", "借款、授信额度最小值"),
        "lmt_avg": ("lmt_avg", "avg(acc_lmt)", "借款、授信额度均值"),
        "last_prd_max": ("last_prd_max", "max(-acc_moi_start)", "首个账户距今"),
        "cls_prd_min": ("cls_prd_min", "min(-cur_moi_closed)", "最后关闭账户距今"),
        "repay_cnt": ("repay_cnt", "sum(PD01ES01)", "近60个月还款记录数"),
        "org_cnt": ("org_cnt", "count(drop_duplicates(PD01AI02))", "机构数"),
        "term_sum": ("term_sum", "sum(PD01ES01)", "还款记录数量"),
        # Last
        "last_repay_mcnt": ("last_repay_mcnt",
                            "min(mixed_doi_last_repay)",
                            "最近还款距今（天）"),
        "last_repay_amt": ("last_repay_amt",
                           "getn(mixed_last_repay_amt, argmin(mixed_doi_last_repay))",
                           "最近还款金额"),
        # Org
        "last_org_cat": ("last_org_cat",
                         "max(argmaxs(acc_doi_start, acc_org_cat))",
                         "最近账户机构类型"),
        "fst_org_cat": ("fst_org_cat",
                        "max(argmins(acc_doi_start, acc_org_cat))",
                        "最早账户机构类型"),
        # Mixed or Infered
        "mixed_ots_sum": ("ots_sum", "sum(mixed_ots)", "剩余额度之和"),
        "mixed_ots_max": ("ots_max", "max(mixed_ots)", "剩余额度最大值"),
        "mixed_folw_monthly_repay_sum": ("mixed_folw_monthly_repay_sum",
                                         "sum(mixed_folw_monthly_repay)",
                                         "按月应还款之和"),
        # D1R41账户
        "folw_prd_max": ("folw_prd_max", "max(acc_moi_end)", "账户预期剩余最长期限"),
        "folw_prd_sum": ("folw_prd_sum", "sum(acc_moi_end)", "账户预期剩余期限和"),
        "alle_prd_max": ("alle_prd_max", "max(acc_moi_range)", "账户最长期限"),
        "alle_prd_sum": ("alle_prd_sum", "sum(acc_moi_range)", "账户期限和"),
        "m2_ovd_pri_sum": ("m2_ovd_pri_sum", "sum(PD01CJ07)", "m2逾期本金之和"),
        "m2_ovd_pri_max": ("m2_ovd_pri_max", "max(PD01CJ07)", "m2逾期本金最大值"),
        "m3_ovd_pri_sum": ("m3_ovd_pri_sum", "sum(PD01CJ08)", "m3逾期本金之和"),
        "m3_ovd_pri_max": ("m3_ovd_pri_max", "max(PD01CJ08)", "m3逾期本金最大值"),
        "m46_ovd_pri_sum": ("m46_ovd_pri_sum", "sum(PD01CJ09)", "m46逾期本金之和"),
        "m46_ovd_pri_max": ("m46_ovd_pri_max", "max(PD01CJ09)", "m46逾期本金最大值"),
        "m7p_ovd_pri_sum": ("m7p_ovd_pri_sum", "sum(PD01CJ10)", "m7p逾期本金之和"),
        "m7p_ovd_pri_max": ("m7p_ovd_pri_max", "max(PD01CJ10)", "m7p逾期本金最大值"),
        # D1R412
        "ovd_sum": ("ovd_sum", "sum(PD01CJ06)", "逾期总额之和"),
        "ovd_max": ("ovd_max", "max(PD01CJ06)", "逾期总额最大值"),
        "monthly_folw_prd_max": ("monthly_folw_prd_max",
                                 "max(PD01CS01)",
                                 "剩余还款期数最大值"),
        "monthly_ovd_prd_sum": ("monthly_ovd_prd_sum",
                                "sum(PD01CS02)",
                                "当前逾期期数之和"),
        "monthly_ovd_prd_max": ("monthly_ovd_prd_max",
                                "max(PD01CS02)",
                                "当前逾期期数最大值"),
        # R2
        "monthly_usd_sum": ("monthly_usd_sum", "sum(PD01CJ02)", "月度已用额度之和"),
        "monthly_usd_max": ("monthly_usd_max", "max(PD01CJ02)", "月度已用额度最大值"),
        "monthly_special_insts_sum": ("monthly_special_insts_sum",
                                      "sum(PD01CJ03)",
                                      "月度未出单大额专项余额之和"),
        "monthly_special_insts_max": ("monthly_special_insts_max",
                                      "max(PD01CJ03)",
                                      "月度未出单大额专项余额最大值"),
        # R3
        "m7p_ovd_sum": ("m7p_ovd_sum", "sum(PD01CJ11)", "m7p透支未付余额之和"),
        "m7p_ovd_max": ("m7p_ovd_max", "sum(PD01CJ11)", "m7p透支未付余额最大值"),
        # R23
        "last_6m_avg_usd_sum": ("last_6m_avg_usd_sum",
                                "sum(last_6m_avg_usd)",
                                "最近6个月平均使用额度之和"),
        "last_6m_avg_usd_max": ("last_6m_avg_usd_max",
                                "sum(last_6m_avg_usd)",
                                "最近6个月平均使用额度最大值"),
        "last_6m_max_usd_max": ("last_6m_max_usd_max",
                                "sum(last_6m_max_usd)",
                                "最近6个月最大使用额度最大值"),
    },
    "cros": [
        # 机构划分
        (["cnt", "lmt_sum", "lmt_max", "mixed_ots_sum", "mixed_ots_max",
          "last_prd_max",
          "cls_prd_min",
          "mixed_folw_monthly_repay_sum"],
         ["orgs", "mois_start"]),
        (["cnt", "lmt_sum", "lmt_max", "mixed_ots_sum", "mixed_ots_max",
          "mixed_folw_monthly_repay_sum"],
         ["orgs", "mois_end"]),
        (["cnt", "lmt_sum", "lmt_max", "mixed_ots_sum", "mixed_ots_max",
          "mixed_folw_monthly_repay_sum"],
         ["orgs", "mixed_acc_status"]),
        # 业务类型划分
        (["cnt", "lmt_sum", "lmt_max", "mixed_ots_sum", "mixed_ots_max",
          "last_prd_max",
          "cls_prd_min",
          "mixed_folw_monthly_repay_sum"],
         ["biz_cat", "mois_start"]),
        (["cnt", "lmt_sum", "lmt_max", "mixed_ots_sum", "mixed_ots_max",
          "last_prd_max",
          "mixed_folw_monthly_repay_sum"],
         ["biz_cat", "mois_end"]),
        (["cnt", "lmt_sum", "lmt_max", "mixed_ots_sum", "mixed_ots_max",
          "last_prd_max",
          "mixed_folw_monthly_repay_sum"],
         ["biz_cat", "mixed_acc_status"]),
        # 所有类型账户
        (["cnt", "repay_cnt", "org_cnt",
          "lmt_sum", "lmt_max", "lmt_min", "lmt_avg",
          "mixed_ots_sum", "mixed_ots_max",
          "last_prd_max", "last_repay_mcnt", "last_repay_amt",
          "term_sum",
          "last_org_cat", "fst_org_cat"],
         ["acc_cat", "mixed_acc_status", "mois_start"]),
        (["cnt", "repay_cnt", "org_cnt",
          "lmt_sum", "lmt_max", "lmt_min", "lmt_avg",
          "mixed_ots_sum", "mixed_ots_max"],
         ["acc_cat", "mixed_acc_status", "guar_type"]),
        # C1账户
        (["cnt", "lmt_sum", "lmt_max", "lmt_min", "last_prd_max"],
         ["acc_cat_c", "trans_status"]),
        # 非C1账户
        (["mixed_folw_monthly_repay_sum"],
         ["acc_cat_nc", "mixed_acc_status"]),
        # D1R41账户
        (["m2_ovd_pri_sum", "m2_ovd_pri_max",
          "m3_ovd_pri_sum", "m3_ovd_pri_max",
          "m46_ovd_pri_sum", "m46_ovd_pri_max",
          "m7p_ovd_pri_sum", "m7p_ovd_pri_max",
          "folw_prd_max", "folw_prd_sum",
          "alle_prd_max", "alle_prd_sum"],
         ["acc_cat_dr41", "mixed_acc_status"]),
        (["cnt", "repay_cnt", "org_cnt",
          "lmt_sum", "lmt_max", "lmt_min", "lmt_avg",
          "mixed_ots_sum", "mixed_ots_max"],
         ["acc_cat", "mixed_lvl5_status"]),
        # D1R412账户
        (["ovd_sum", "ovd_max",
         "monthly_folw_prd_max", "monthly_ovd_prd_sum", "monthly_ovd_prd_max"],
         ["acc_cat_dr412"]),
        # R2
        (["monthly_usd_sum", "monthly_usd_max",
          "monthly_special_insts_sum", "monthly_special_insts_max"],
         ["acc_cat_r2", "mixed_acc_status"]),
        (["cnt", ],
         ["acc_cat_r2", "mixed_acc_status", "usd_ppor"]),
        # R23
        (["last_6m_avg_usd_sum", "last_6m_avg_usd_max", "last_6m_max_usd_max"],
         ["acc_cat_r23", "mixed_acc_status"]),
    ],
}


# %%
CREDIT_INFO = {
    "part": "credit_info",
    "desc": "报文授信协议",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_credit_info"],
    "joinkey": None,
    "key_fmt": "credit_{cond}_{agg}",
    "cond": {
        "orgs": org_cat("credit_org_cat"),
        "mois_start": last_mois("credit_moi_start", "开立"),
        "mois_end": folw_mois("credit_moi_end"),
        "credit_cat": credit_cat("credit_cat"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "授信协议数量"),
        "lmt_sum": ("lmt_sum", "sum(PD02AJ01)", "授信额度之和"),
        "lmt_max": ("lmt_max", "max(PD02AJ01)", "授信额度最大值"),
        "lmt_min": ("lmt_min", "max(PD02AJ01)", "授信额度最小值"),
        "lmt2_sum": ("lmt2_sum", "sum(PD02AJ03)", "授信限额之和"),
        "lmt2_max": ("lmt2_max", "max(PD02AJ03)", "授信限额最大值"),
        "usd_sum": ("usd_sum", "sum(PD02AJ04)", "已用额度之和"),
        "usd_max": ("usd_max", "max(PD02AJ04)", "已用额度最大值"),
    },
    "cros": [
        (["cnt", "lmt_sum", "lmt_max", "lmt_min",
          "lmt2_sum", "lmt2_max", "usd_sum", "usd_max"],
         ["orgs", "mois_start"]),
        (["cnt", "lmt_sum", "lmt_max", "lmt_min",
          "lmt2_sum", "lmt2_max", "usd_sum", "usd_max"],
         ["orgs", "mois_end"]),
        (["cnt", "lmt_sum", "lmt_max", "lmt_min",
          "lmt2_sum", "lmt2_max", "usd_sum", "usd_max"],
         ["credit_cat", "mois_start"]),
        (["cnt", "lmt_sum", "lmt_max", "lmt_min",
          "lmt2_sum", "lmt2_max", "usd_sum", "usd_max"],
         ["credit_cat", "mois_end"]),
    ]
}


# %%
REL_INFO = {
    "part": "rel_info",
    "desc": "报文相关还款责任",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_rel_info"],
    "joinkey": None,
    "key_fmt": "rel_{cond}_{agg}",
    "cond": {
        "orgs": org_cat("rel_org_cat"),
        "biz_cat": biz_cat("rel_biz_cat"),
        "lvl5_status": lvl5_status("rel_lvl5_status"),
        "repay_status": repay_status("rel_repay_status"),
        "mois_start": last_mois("rel_moi_start", "开立"),
        "mois_end": folw_mois("rel_moi_end"),
        "ovd_month": amt_threshs_1("ovd_prd", "PD03AS01", "逾期月数", 7),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "相关还款责任数"),
        "respon_sum": ("resp_sum", "sum(PD03AJ01)", "相关责任金额之和"),
        "respon_max": ("resp_max", "max(PD03AJ01)", "相关责任金额最大值"),
        "acc_sum": ("acc_sum", "sum(PD03AJ02)", "相关责任账户金额之和"),
        "acc_max": ("acc_max", "max(PD03AJ02)", "相关责任账户金额最大值"),
        "repay_status_max": ("repay_status_max", "max(rel_repay_status)",
                             "相关责任逾期月数最大值"),
        "repay_status_sum": ("repay_status_sum", "sum(rel_repay_status)",
                             "相关责任逾期月数之和"),
        "ovd_month_max": ("ovd_month_max", "max(PD03AS01)",
                          "相关责任逾期月数最大值"),
        "ovd_month_sum": ("ovd_month_sum", "sum(PD03AS01)",
                          "相关责任逾期月数之和"),
        "folw_prd_max": ("folw_prd_max", "max(rel_moi_end)",
                         "相关责任剩余最长期限"),
        "last_prd_max": ("last_prd_max", "max(-rel_moi_start)",
                         "相关责任首个账户距今"),
    },
    "cros": [
        (["cnt", "respon_sum", "respon_max", "acc_sum", "acc_max"],
         ["biz_cat", ]),
        (["cnt", "respon_sum", "respon_max", "acc_sum", "acc_max"],
         ["orgs", ]),
        (["cnt", "respon_sum", "respon_max", "acc_sum", "acc_max"],
         ["lvl5_status", ]),
        (["cnt", "respon_sum", "respon_max", "acc_sum", "acc_max"],
         ["repay_status", ]),
        (["cnt", "respon_sum", "respon_max", "acc_sum", "acc_max"],
         ["ovd_month", ]),
        (["cnt", "respon_sum", "respon_max", "acc_sum", "acc_max"],
         ["mois_start", ]),
        (["cnt", "respon_sum", "respon_max", "acc_sum", "acc_max"],
         ["mois_end", ]),
        (["repay_status_sum", "repay_status_max",
          "ovd_month_sum", "ovd_month_max",
          "folw_prd_max", "last_prd_max"],
         ["biz_cat", ]),
    ],
}


# %%
POSTFEE_INFO = {
    "part": "postfee_info",
    "desc": "报文后付费信息",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_postfee_info"],
    "joinkey": None,
    "key_fmt": "postfee_{cond}_{agg}",
    "cond": {
        "mois": last_mois("mon_itvl(PE01AR02, today)"),
        "acc_cat": postfee_acc_cat("postfee_acc_cat"),
        "acc_status": postfee_acc_status("postfee_acc_status"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "账户数"),
        "ovd_sum": ("ovd_sum", "sum(PE01AJ01)", "欠费金额之和"),
        "ovd_max": ("ovd_max", "max(PE01AJ01)", "欠费金额最大值"),
    },
    "cros": [
        (["cnt", ]                          , ["acc_cat", "acc_status"]),
        (["ovd_sum", "ovd_max"]             , ["acc_cat", "mois"]),
    ],
}


# %%
TAXS = {
    "part": "taxs",
    "desc": "报文欠税记录",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_taxs"],
    "joinkey": None,
    "key_fmt": "tax_{cond}_{agg}",
    "cond": {
        "mois": last_mois("mon_itvl(PF01AR01, today)")
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "账户数"),
        "ovd_sum": ("ovd_sum", "sum(PF01AJ01)", "欠税额之和"),
        "ovd_max": ("ovd_max", "max(PF01AJ01)", "欠税额最大值"),
    },
    "cros": [
        (["cnt", "ovd_sum", "ovd_max"]  , ["mois", ]),
    ],
}


# %%
LAWSUIT = {
    "part": "lawsuit",
    "desc": "报文涉诉记录",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_lawsuit"],
    "joinkey": None,
    "key_fmt": "lawsuit_{cond}_{agg}",
    "cond":{
        "mois": last_mois("mon_itvl(PF02AR01, today)"),
        "None": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "民事判决案件数"),
        "target_sum": ("target_sum", "sum(PF02AJ01)", "民事判决标的金额之和"),
        "target_max": ("target_max", "max(PF02AJ01)", "民事判决标的金额最大值"),
        "prd_max": ("prd_max", "max(mon_itvl(today, PF02AR01))", "最早民事判决距今"),
        "prd_min": ("prd_min", "min(mon_itvl(today, PF02AR01))", "最晚民事判决距今"),
    },
    "cros": [
        (["cnt", "target_sum", "target_max"],
         ["mois"]),
        (["prd_max", "prd_min"],
         ["None"]),
    ],
}


# %%
ENFORCEMENT = {
    "part": "enforcement",
    "desc": "报文被执行",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_enforcement"],
    "joinkey": None,
    "key_fmt": "enforcement_{cond}_{agg}",
    "cond": {
        "mois": last_mois("mon_itvl(PF03AR01, today)"),
        "None": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "强制执行案件数"),
        "todo_sum": ("todo_sum", "sum(PF03AJ01)", "申请执行标的金额之和"),
        "todo_max": ("todo_max", "max(PF03AJ01)", "申请执行标的金额最大值"),
        "done_sum": ("done_sum", "sum(PF03AJ02)", "已执行标的金额之和"),
        "done_max": ("done_max", "max(PF03AJ02)", "已执行标的金额最大值"),
        "prd_max": ("prd_max", "max(mon_itvl(today, PF03AR01))", "最早强制执行距今"),
        "prd_min": ("prd_min", "min(mon_itvl(today, PF03AR01))", "最晚强制执行距今"),
    },
    "cros": [
        (["cnt", "todo_sum", "todo_max", "done_sum", "done_max"],
         ["mois",]),
        (["prd_max", "prd_min"],
         ["None"]),
    ],
}


# %%
GOV_PUNISHMENT = {
    "part": "gov_punishment",
    "desc": "报文行政处罚",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_gov_punishment"],
    "joinkey": None,
    "key_fmt": "gov_punishment_{cond}_{agg}",
    "cond": {
        "mois": last_mois("mon_itvl(PF04AR01, today)"),
        "None": none_of_all(),

    },
    "agg": {
        "cnt": ("cnt", "count(_)", "行政处罚数量"),
        "penalty_sum": ("sum", "sum(PF04AJ01)", "行政处罚金额之和"),
        "penalty_max": ("max", "max(PF04AJ01)", "行政处罚金额最大值"),
        "prd_max": ("prd_max", "max(mon_itvl(today, PF04AR01))", "最早行政处罚距今"),
        "prd_min": ("prd_min", "min(mon_itvl(today, PF04AR01))", "最晚行政处罚距今"),
    },
    "cros": [
        (["cnt", "penalty_sum", "penalty_max"]  , ["mois"]),
        (["prd_max", "prd_min"]                 , ["None"]),
    ],
}


# %%
HOUSING_FUND = {
    "part": "housing_fund",
    "desc": "报文住房公积金",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_housing_fund"],
    "joinkey": None,
    "key_fmt": "housing_fund_{cond}_{agg}",
    "cond": {
        "mois_end": last_mois("mon_itvl(PF05AR03, today)", "缴交"),
        "hf_status": housing_fund_status("hf_status"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "住房公积金账户数量"),
        "sum": ("sum", "sum(PF05AJ01)", "住房公积金月缴存额之和"),
        "max": ("max", "max(PF05AJ01)", "住房公积金月缴存额最大值"),
        "start_prd_max": ("start_prd_max",
                          "max(mon_itvl(today, PF05AR02))",
                          "最早初缴距今"),
        "start_prd_min": ("start_prd_min",
                          "min(mon_itvl(today, PF05AR02))",
                          "最晚初缴距今"),
        "end_prd_max": ("end_prd_max",
                        "max(mon_itvl(today, PF05AR03))",
                        "最晚缴至距今"),
        "end_prd_min": ("end_prd_min",
                        "min(mon_itvl(today, PF05AR03))",
                        "最早缴至距今"),
        "latest_prd_min": ("latest_prd_min",
                           "min(mon_itvl(today, PF05AR04))",
                           "最近缴交距今"),
    },
    "cros": [
        (["cnt", "sum", "max"]                  , ["hf_status"]),
        (["cnt", "sum", "max"]                  , ["mois_end"]),
        (["start_prd_max", "start_prd_min",
          "end_prd_max", "end_prd_min",
          "latest_prd_min"],
         ["hf_status"])
    ],
}


# %%
SUB_ALLOWANCE = {
    "part": "sub_allowance",
    "desc": "报文低保",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_sub_allowance"],
    "joinkey": None,
    "key_fmt": "allowance_{cond}_{agg}",
    "cond": {
        "mois": last_mois("mon_itvl(PF06AR01, today)")
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "低保救助记录数量"),
    },
    "cros": [
        (["cnt",]                       , ["mois",]),
    ],
}


# %%
PRO_CERT = {
    "part": "pro_cert",
    "desc": "报文职业证照",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_pro_cert"],
    "joinkey": None,
    "key_fmt": "cert_{cond}_{agg}",
    "cond": {
        "mois": last_mois("mon_itvl(PF07AR01, today)"),
        "None": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "职业资格数量"),
    },
    "cros": [
        (["cnt",]                       , ["None",]),
    ],
}


# %%
GOV_AWARD = {
    "part": "gov_award",
    "desc": "报文政府奖励",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_gov_award"],
    "joinkey": None,
    "key_fmt": "award_{cond}_{agg}",
    "cond": {
        "mois": last_mois("mon_itvl(PF08AR01, today)"),
        "None": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "奖励数量"),
    },
    "cros": [
        (["cnt",]                       , ["None",]),
    ],
}


# %%
INQ_REC = {
    "part": "inq_rec",
    "desc": "报文查询",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_inq_rec"],
    "joinkey": None,
    "key_fmt": "inq_rec_{cond}_{agg}",
    "cond": {
        "mois": last_mois("inq_rec_moi") + last_dois("inq_rec_doi"),
        # 查询模块中，`org_bank` 上线字段中意为 “银行机构”
        # 但账户模块中，`org_bank` 上线字段中意为 “商业银行”
        "orgs": (org_cat("inq_rec_org_cat")
                 + [("org_bank", "inq_rec_org_cat < 20", "银行机构")]),
        "inq_reason": inq_rec_reason_cat("inq_rec_reason_cat")
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "查询数量"),
        "latest_prd_min": ("latest_prd_min", "min(-inq_rec_moi)", "距今月份数最小值"),
        "org_cnt": ("org_cnt", "count(drop_duplicates(PH010Q02))", "机构数"),
        "org_cat_cnt": ("org_cat_cnt", "count(drop_duplicates(inq_rec_org_cat))",
                        "机构类型数"),
        "org_reason_day_cnt": ("org_reason_cnt",
                               "count(drop_duplicates([PH010R01, PH010Q02, PH010Q03]))",
                               "机构、日期、查询原因数"),
        "last_6in24m_coef_var": ("coef_var",
                                 "coef_var(hist(inq_rec_moi, [-24, -18, -12, -6, 0]))",
                                 "近24个月查询量变异系数"),
    },
    "cros": [
        (["cnt", "org_cnt", "org_cat_cnt", "org_reason_day_cnt",
          "latest_prd_min"],
         ["mois", "orgs", "inq_reason"]),
        (["last_6in24m_coef_var",],
         ["orgs", "inq_reason",]),
    ],
}


# %%
LV1_AGG_CONF = {
    PINFO_MOBILE["part"]            : PINFO_MOBILE,
    PINFO_RES["part"]               : PINFO_RES,
    PINFO_COMP["part"]              : PINFO_COMP,
    ACC_INFO["part"]                : ACC_INFO,
    CREDIT_INFO["part"]             : CREDIT_INFO,
    REL_INFO["part"]                : REL_INFO,
    POSTFEE_INFO["part"]            : POSTFEE_INFO,
    TAXS["part"]                    : TAXS,
    LAWSUIT["part"]                 : LAWSUIT,
    ENFORCEMENT["part"]             : ENFORCEMENT,
    GOV_PUNISHMENT["part"]          : GOV_PUNISHMENT,
    HOUSING_FUND["part"]            : HOUSING_FUND,
    SUB_ALLOWANCE["part"]           : SUB_ALLOWANCE,
    PRO_CERT["part"]                : PRO_CERT,
    GOV_AWARD["part"]               : GOV_AWARD,
    INQ_REC["part"]                 : INQ_REC,
}


# %%
R2ACC_WITH_CDT = {
    "part": "r2acc_with_cdt",
    "desc": "信用卡账户与授信协议",
    "level": 1,
    "prikey": ["rid", "certno", "PD01AI04"],
    "from_": ["pboc_acc_info", "pboc_credit_info"],
    "joinkey": [["rid", "certno", "PD01AI04"],
                ["rid", "certno", "PD02AI03"]],
    "key_fmt": "r2acc_with_cdt_{cond}_{agg}",
    "cond": {
        "fixed_cond": [(None, "notnull(PD01AI04)", "R2授信协议下")],
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "账户数"),
        "lmt": ("lmt", "max(PD02AJ01)", "授信额度"),
        "usd": ("usd", "max(PD02AJ04)", "已用额度"),
        "monthly_usd": ("monthly_ots", "sum(PD01BJ01)", "各账户余额之和"),
        "monthly_ots": ("monthly_usd",
                        "sum(PD01CJ02)",
                        "各账户月度已用额度之和"),
        "monthly_special_insts": ("monthly_special_insts",
                                  "sum(PD01CJ03)",
                                  "各账户月度未出单大额专项余额之和"),
        "last_6m_avg_usd": ("last_6m_avg_usd",
                            "sum(last_6m_avg_usd)",
                            "各账户最近6个月平均使用额度之和"),
        "last_6m_max_usd": ("last_6m_max_usd",
                            "sum(last_6m_max_usd)",
                            "各账户最近6个月最大使用额度秩和"),
        "status": ("status", "max(mixed_acc_status)", "最差账户状态"),
        "moi_start": ("moi_start", "min(acc_moi_start)", "最早账户距今月"),
    },
    "cros": [
        (["cnt", "lmt", "usd", "status", "moi_start",
          "monthly_ots", "monthly_usd", "monthly_special_insts",
          "last_6m_avg_usd", "last_6m_max_usd"],
         ["fixed_cond"]),
    ],
}


# %%
R2ACC_WITH_CDT_AGG = {
    "part": "r2acc_with_cdt_agg",
    "desc": "报文信用卡账户与授信协议",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["r2acc_with_cdt"],
    "joinkey": None,
    "key_fmt": "{cond}_{agg}",
    "cond": {
        "None": none_of_all(),
    },
    "agg": _basic_upper_aggs(R2ACC_WITH_CDT),
    "cros": [
        (list(_basic_upper_aggs(R2ACC_WITH_CDT).keys()),
         ["None"]),
    ],
}


# %%
R2ACC_WITH_ORG = {
    "part": "r2acc_with_org",
    "desc": "信用卡账户分机构",
    "level": 1,
    "prikey": ["rid", "certno", "PD01AI02"],
    "from_": ["pboc_acc_info"],
    "joinkey": None,
    "key_fmt": "r2acc_with_org_{cond}_{agg}",
    "cond": {
        "fixed_cond": [(None, "isnull(PD01AI04)", "R2机构下")],
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "账户数"),
        "lmt": ("lmt", "sum(PD01AJ03)", "授信额度"),
        "monthly_usd": ("monthly_ots_sum", "sum(PD01BJ01)", "各账户余额之和"),
        "monthly_ots": ("monthly_usd_sum", "sum(PD01CJ02)", "各账户月度已用额度之和"),
        "monthly_special_insts": ("monthly_special_insts_sum",
                                  "sum(PD01CJ03)",
                                  "各账户月度未出单大额专项余额之和"),
        "last_6m_avg_usd": ("last_6m_avg_usd_sum",
                            "sum(last_6m_avg_usd)",
                            "各账户最近6个月平均使用额度之和"),
        "last_6m_max_usd": ("last_6m_max_usd_sum",
                            "sum(last_6m_max_usd)",
                            "各账户最近6个月最大使用额度之和"),
        "status": ("status", "max(mixed_acc_status)", "最差账户状态"),
        "moi_start": ("moi_start", "min(acc_moi_start)", "最早账户距今月"),
    },
    "cros": [
        (["cnt", "lmt", "status", "moi_start",
          "monthly_ots", "monthly_usd", "monthly_special_insts",
          "last_6m_avg_usd", "last_6m_max_usd"],
         ["fixed_cond"]),
    ],
}


# %%
R2ACC_WIHT_ORG_AGG = {
    "part": "r2acc_with_org_agg",
    "desc": "信用卡账户分机构聚集",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["r2acc_with_org"],
    "joinkey": None,
    "key_fmt": "{cond}_{agg}",
    "cond": {
        "None": none_of_all(),
    },
    "agg": _basic_upper_aggs(R2ACC_WITH_ORG),
    "cros": [
        (list(_basic_upper_aggs(R2ACC_WITH_ORG).keys()),
         ["None"]),
    ],
}


# %%
R2ACC_GROUP_CONF = {
    R2ACC_WITH_CDT["part"]: R2ACC_WITH_CDT,
    R2ACC_WITH_ORG["part"]: R2ACC_WITH_ORG,
}

R2ACC_GROUP_AGG_CONF = {
    R2ACC_WITH_CDT_AGG["part"]: R2ACC_WITH_CDT_AGG,
    R2ACC_WIHT_ORG_AGG["part"]: R2ACC_WIHT_ORG_AGG,
}


# %%
def df_agg_confs(confs: dict = None):
    import pandas as pd
    if confs is None:
        confs = {**LV2_AGG_CONF, **LV20_AGG_CONF,
                 **LV1_AGG_CONF,
                 **R2ACC_GROUP_CONF, **R2ACC_GROUP_AGG_CONF}
    pconfs = []
    aconfs = {}
    for pname, pconf in confs.items():
        pname = pconf["part"]
        pconfs.append((pconf["part"],
                       pconf["desc"],
                       pconf["level"],
                       pconf["prikey"],
                       pconf["from_"],
                       pconf["joinkey"]))
        aconf = cross_aggs_and_filters(
            pconf["cros"], pconf["agg"], pconf["cond"], pconf["key_fmt"])
        aconfs[pname] = pd.DataFrame.from_records(
            aconf, columns=["key", "cond", "agg", "cmt"])

    # Concat the confs.
    pconfs = pd.DataFrame.from_records(
        pconfs, columns=["part", "desc", "level", "prikey", "from_", "joinkey"])
    aconfs = pd.concat(aconfs.values(), keys=aconfs.keys()).droplevel(level=1)
    aconfs.index.set_names("part", inplace=True)
    aconfs = aconfs.reset_index()

    return pconfs, aconfs
