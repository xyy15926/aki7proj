#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: confmark.py
#   Author: xyy15926
#   Created: 2024-11-15 14:31:30
#   Updated: 2024-12-09 19:14:22
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd
from itertools import product, chain


# %%
MARKS_INQ = {
    "part": "pboc_inq_marks",
    "desc": "人行查询",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["pboc_inq_abst", "inq_rec", "acc_info"],
    "joinkey": [["rid", "certno"],
                ["rid", "certno"],
                ["rid", "certno"]],
    "trans": [
        # # 多头申请_人行历史：待删除
        # ("AAZA1001", "", None, "多头申请_人行历史"),
        # # 多头申请_人行2年内
        # ("AAZB1001", "", None, "多头申请_人行近2年"),
        # 多头申请_人行1年内
        ("AAZC1001", "inq_rec_last_12m_for_loan_cnt > 30", None, "人行近12月贷款审批记录异常"),
        # 多头申请_人行6月内
        ("AAZD1001", "inq_rec_last_6m_for_prev3_cnt > 25", None, "近6月信用卡、贷款、融资查询异常"),
        # 多头申请_人行3月内
        ("AAZE1001", "inq_rec_last_3m_for_loan_cnt > 12", None, "人行近3月贷款审批记录异常"),
        ("AAZE1002", "inq_rec_last_1m_for_prev3_cnt > 15", None, "近1月信用卡、贷款、融资查询异常"),
        ("AAZE1002", "(inq_rec_last_2m_for_loan_cnt > 8) & (acc_d1r41_last_2m_cnt <= 0)", None, "人行近2月贷款审批且批核异常"),
        ("AAZE1003", "(inq_rec_last_3m_cnt > 3) & (acc_last_prd_max < 6)", None, "白户且近3月机构查询异常"),
        ("AAZE1004", "(acc_lmt_max < 5000) & (inq_rec_last_3m_cnt > 3)", None, "低额度且近3月机构查询异常"),
        # 多头申请_人行1月内
        ("AAZF1001", "(PC05BS05 > 3) & (acc_last_prd_max < 6)", None, "白户且近1月个人自查异常"),
        ("AAZF1002", "inq_rec_last_1m_for_prev3_cnt > 10", None, "近1月信用卡、贷款、融资查询异常"),
        ("AAZF1003", "sadd(inq_rec_last_2d_dorg_sloan_for_loan_cnt, inq_rec_last_2d_dorg_sloan_for_loan_cnt) > 0", None, "近2日贷款公司、小额贷公司贷款查询异常"),
        ("AAZF1004", "inq_rec_last_2d_dorg_autofin_cnt > 1", None, "近2日汽金机构查询异常"),
        ("AAZF1005", "inq_rec_last_2d_dorg_leasing_cnt > 1", None, "近2日融租机构查询异常"),
        ("AAZF1006", "inq_rec_last_2d_for_loan_cnt > 5", None, "近2日贷款查询异常"),
    ],
}


# %%
MARKS_ACC = {
    "part": "pboc_acc_marks",
    "desc": "人行账户",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["acc_info"],
    "joinkey": None,
    "trans": [
        # 账户数量_人行
        ("ACZA1001", "acc_biz_cat_auto_last_6m_cnt > 1", None, "近6月新增车贷异常"),
        ("ACZA1002", "acc_biz_cat_auto_nocls_cnt > 2", None, "未结清车贷"),
        # # 负债水平_人行
        # ("ACZB1001", "", None, "负债水平_人行"),
        # 信贷记录短_人行白户
        ("ACZC1001", "acc_last_prd_max < 6", None, "信贷记录短"),
    ],
}


# %%
MARKS_OVD = {
    "part": "pboc_ovd_marks",
    "desc": "人行逾期",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": ["acc_info", "acc_repay_60m_agg_basic"],
    "joinkey": [["rid", "certno"],
                ["rid", "certno"]],
    "trans": [
        # 人行逾期_历史
        ("AOZA1001", "d1r41acc_repay_eq7_cnt_sum > 0", None, "贷款历史存在7"),
        ("AOZA1002", "d1r41acc_repay_gpaid_cnt_sum > 0", None, "贷款历史担保人代还"),
        ("AOZA1003", "r2acc_repay_gpaid_cnt_sum > 0", None, "信用卡历史担保人代还"),
        # 人行逾期_2年内
        ("AOZB1001", "(d1r41acc_repay_last_24m_le1_cnt_max > 20) & (d1r41acc_repay_last_24m_ovd_max_max > 1000)", None, "单贷款近2年累计逾期"),
        ("AOZB1002", "(r2acc_repay_last_24m_le1_cnt_max > 20) & (r2acc_repay_last_24m_ovd_max_max > 1000)", None, "单信用卡近2年累计逾期"),
        ("AOZB1003", "d1r41acc_repay_last_24m_status_ovd_conl_max_max > 3", None, "单贷款近2年连续逾期"),
        ("AOZB1004", "r2acc_repay_last_24m_status_ovd_conl_max_max > 3", None, "单信用卡近2年连续逾期"),
        ("AOZB1005", "(acc_repay_last_24m_le1_cnt_sum > 30) & (acc_repay_last_24m_ovd_max_max > 1000)", None, "多账户近2年累计逾期"),
        # 人行逾期_1年内
        ("AOZC1001", "d1r41acc_repay_last_12m_le1_cnt_max > 6", None, "单贷款近1年累计逾期"),
        ("AOZC1002", "r2acc_repay_last_12m_le1_cnt_max > 6", None, "单信用卡近1年累计逾期"),
        ("AOZC1003", "d1r41acc_repay_last_12m_status_ovd_conl_max_max > 3", None, "单贷款近1年连续逾期"),
        ("AOZC1004", "r2acc_repay_last_12m_status_ovd_conl_max_max > 3", None, "单信用卡近1年连续逾期"),
        ("AOZC1005", "d1r41acc_repay_last_12m_status_max_max > 2", None, "贷款近1年最大逾期期数"),
        ("AOZC1006", "r2acc_repay_last_12m_status_max_max > 4", None, "信用卡近1年最大逾期期数"),
        # # 人行逾期_6月内
        # ("AOZD1001", "", None, "人行逾期_6月内"),
        # 人行逾期_3月内
        ("AOZE1001", "(d1r41acc_repay_last_3m_status_max_max > 0) & (d1r41acc_repay_last_24m_ovd_max_max > 1000)", None, "贷款近3月逾期"),
        # 人行逾期_当前
        ("AOZF1001", "acc_r2_ovd_max > 500", None, "信用卡当前逾期"),
        ("AOZF1002", "acc_d1r41_ovd_max > 500", None, "贷款当前逾期"),
        # 人行逾期_类逾期记录
        ("AOZG1001", "sadd(acc_abnor_cnt, acc_dum_cnt) > 0", None, "异常、呆账账户"),
        ("AOZG1002", "acc_d1r41_lvl5_inf_cnt > 0", None, "贷款五级分类异常"),
        ("AOZG1003", "acc_c1_cnt > 0", None, "追偿账户"),
    ],
}


# %%
_MARKS_INFO = [
    # 关联手机号异常_人行
    ("pinfo_mobile", [
        ("ARZA1001", "pinfo_mobile_last_6m_cnt > 3", None, "近6月关联手机号异常"),
    ]),
    # # 申请人居住地限制
    # ("pinfo_res", [
    #     ("ADZA1001", "", None, "申请人居住地限制_人行"),
    # ]),
    # # 申请人职业限制_人行
    # ("pinfo_comp", [
    #     ("AJZA1001", "", None, "申请人职业限制_人行"),
    # ]),
    # 法院被执行记录_人行
    ("enforcement", [
        ("APZA1001", "enforcement_cnt > 0", None, "人行法院被执行记录"),
    ]),
    # 法院判决记录_人行
    ("lawsuit", [
        ("APZB1001", "lawsuit_cnt > 0", None, "人行法院判决记录"),
    ]),
    # 行政处罚记录_人行
    ("gov_punishment", [
        ("APZC1001", "gov_punishment_cnt > 0", None, "人行行政处罚记录"),
    ]),
    # 非信贷欠费_人行
    ("postfee_info", [
        ("APZD1001", "postfee_ovd_sum > 0", None, "人行非信贷欠费记录"),
    ]),
    ("taxs", [
        ("APZD1002", "tax_ovd_max > 0", None, "人行欠税记录"),
    ]),
    # 低保救助_人行
    ("sub_allowance", [
        ("APZE1001", "allowance_cnt > 0", None, "人行低保救助记录"),
    ]),
]

_MARKS_INFO_TMPL = {
    pname: {
        "part": f"pboc_info_marks_{pname}",
        "desc": f"人行基本信息{pname}",
        "level": 0,
        "prikey": ["rid", "certno"],
        "from_": [pname],
        "joinkey": None,
        "trans": trans,
    } for pname, trans in _MARKS_INFO
}


# %%
MARKS_INFO = {
    "part": "pboc_info_marks",
    "desc": "人行基本信息",
    "level": 0,
    "prikey": ["rid", "certno"],
    "from_": [p["part"] for p in _MARKS_INFO_TMPL.values()],
    "joinkey": [["rid", "certno"]] * len(_MARKS_INFO_TMPL),
    "trans": list(chain.from_iterable([
        [(key, key, None, desc) for key, trans, cond, desc in p["trans"]]
        for p in _MARKS_INFO_TMPL.values()])),
}

_MARKS_INFO_TMPL[MARKS_INFO["part"]] = MARKS_INFO


# %%
TRANS_CONF = {
    MARKS_INQ["part"]: MARKS_INQ,
    MARKS_ACC["part"]: MARKS_ACC,
    MARKS_OVD["part"]: MARKS_OVD,
    **_MARKS_INFO_TMPL,
}


# %%
def df_mark_confs():
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
        marks = [(part_name, key, cond, trans, desc)
                 for key, trans, cond, desc in pconf["trans"]]
        tconfs.extend(marks)

    pconfs = pd.DataFrame.from_records(
        pconfs, columns=["part", "desc", "level", "prikey",
                         "from_", "joinkey"])
    tconfs = pd.DataFrame.from_records(
        tconfs, columns=["part", "key", "cond", "trans", "desc"])

    return pconfs, tconfs
