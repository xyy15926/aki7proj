#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: conftrans.py
#   Author: xyy15926
#   Created: 2024-09-23 09:57:58
#   Updated: 2024-11-04 11:10:59
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar
from itertools import product

import pandas as pd
from suitbear.finer import get_assets_path


# %%
def daytag_busi(field: str = "apply_date"):
    reprs = [("busi", f"is_busiday({field})", "工作日"),
             ("nbusi", f"not_busiday({field})", "节假日"),
             (None, None, None)]
    return reprs


def timetag(field: str = "apply_time"):
    reprs = [("earlymor", f"{field} <= 6", "凌晨"),
             ("mor", f"({field} > 6) & ({field} <= 11)", "上午"),
             ("noon", f"({field} > 11 & ({field} <= 14)", "中午"),
             ("afternoon", f"({field} > 14 & ({field} <= 18)", "下午"),
             ("eve", f"({field} > 18 & ({field} <= 24)", "晚间"),
             (None, None, None)]
    return reprs


# %%
REGION_CAT = {
    "36": ("SN", "华南"),
}


def map_gov_region_cat(val: str):
    val = REGION_CAT.get(val)
    if val is None:
        return None
    else:
        return val[0]


def gov_region_cats(field: str = "region_code"):
    reprs = []
    for key, desc in REGION_CAT.values():
        reprs.append((key, f"{field} == key", desc))
    return reprs


# %%
APPROVAL_FLAG = {
    # 申贷多头
    2101: (2101   , "申贷记录异常_人行"),
    2102: (2102   , "申贷记录异常_三方"),
    2103: (2103   , "申贷记录异常_内部"),
    # 公开名单、基本信息核验
    2201: (2201   , "手机号状态异常"),
    2202: (2202   , "基本信息校验异常"),
    2203: (2203   , "命中司法、公安名单"),
    2205: (2205   , "非信贷行为异常_三方"),
    # 信贷、还款
    2301: (2301   , "信贷逾期_人行"),
    2302: (2302   , "信贷逾期_三方"),
    2303: (2303   , "信贷逾期_内部"),
    2304: (2304   , "信贷历史短_人行"),
    2305: (2305   , "信贷负债高_人行"),
    2306: (2306   , "信贷额度低_人行"),
    2307: (2307   , "敏感信贷业务_人行"),
    # 评分
    2401: (2401   , "评分异常_三方"),
    2402: (2402   , "评分异常_内部"),
    # 关联信息
    3101: (3101   , "关联异常_手机号"),
    3102: (3102   , "关联异常_身份证号"),
    3201: (3201   , "关联异常_配偶"),
    3202: (3202   , "关联异常_联系人"),
    3203: (3203   , "关联异常_担保人"),
    3204: (3204   , "关联异常_单位"),
    3301: (3301   , "关联异常_门店"),
    3302: (3302   , "关联异常_业务员"),
    # 标的、车
    4101: (4101   , "车况异常"),
    4102: (4102   , "车辆估值异常"),
    4103: (4103   , "车辆历史交易异常"),
    4104: (4104   , "车辆关联人、机构异常"),
    # 企业
    5101: (5101   , "企业经营异常"),
    5102: (5102   , "企业涉诉、失信"),
    5103: (5103   , "企业高管变动"),
    # 准入条件
    9101: (9101   , "准入条件限制_个人"),
    9201: (9201   , "准入条件限制_法人"),
    9301: (9301   , "准入条件限制_车辆"),
    9401: (9401   , "准入条件限制_融资"),
    # 名单控制
    9501: (9501   , "黑白名单控制"),
    9999: (9999   , "系统配置"),
}


def approval_flag_deprecated(field: str = "rej_cat"):
    reprs = [("manual", f"{field} > 9500", "人工介入"),
             ("apply_behav", f"({field} > 2100) & ({field} < 2200)", "多头申请"),
             ("nloan_behav", f"({field} > 2200) & ({field} < 2300)", "非信贷行为"),
             ("loan_behav", f"({field} > 2300) & ({field} < 2400)", "信贷还款行为"),
             ("score", f"({field} > 2400) & ({field} < 2500)", "综合评分"),
             ("rel_self", f"({field} > 3100) & ({field} < 3200)", "自身关联异常"),
             ("rel_link", f"({field} > 3200) & ({field} < 3300)", "关联人关联异常"),
             ("rel_trade", f"({field} > 3300) & ({field} < 3400)", "交易关联异常"),
             ("car_risk", f"({field} > 4100) & ({field} < 4200)", "车辆异常"),
             ("comp_risk", f"({field} > 5100) & ({field} < 5200)", "公司异常"),
             ("rule_out", f"({field} > 9100) & ({field} < 9500)", "准入条件限制")]
    return reprs


def rej_flag(field: str = "rej_cat"):
    reprs = [("rej_behav", f"({field} > 2000) & ({field} < 3000)", "信用资质拒绝"),
             ("rej_rel", f"({field} > 3000) & ({field} < 4000)", "关联关系拒绝"),
             ("rej_rule", f"{field} > 9000", "准入条件拒绝")]
    return reprs


# %%
APPROVAL_RESULT = {
    "accept": (1        , "通过"),
    "reject": (2        , "拒绝"),
    "validation": (3    , "待定"),
}


def approval_result_reprs(field: str = "approval_result"):
    reprs = [
        ("acp", f'{field} == "accept"', "通过"),
        ("rej", f'{field} == "reject"', "拒绝"),
        ("vld", f'{field} == "validation"', "待定"),
        (None, None, None),
    ]
    return reprs


# %%
def get_infocode_mapper():
    infocode_map = {
        "AF10005": "NLBH",
        "AF10006": "LBH",
        "AF10009": "NLBH",
        "AF10018": "RELS",
        "AF10019": "PRDR",
        "AF10047": "SCR",
        "AF20021": "OTH",
        "AF20003": None,
        "AF20020": "OTH",
        "AF20002": None,
        "AF100047": "",
    }
    return infocode_map


def approval_code_reprs(field: str = "infocode_cats"):
    approval_code_cats = [
        ("manual"           , "MAN"     , "人工介入"),
        ("apply_behav"      , "APBH"    , "多头申请"),
        ("nloan_behav"      , "NLBH"    , "非信贷行为"),
        ("loan_behav"       , "LBH"     , "信贷还款行为"),
        ("score"            , "SCR"     , "综合评分"),
        ("rel_self"         , "RELS"    , "自身关联异常"),
        ("rel_link"         , "RELL"    , "关联人关联异常"),
        ("rel_trade"        , "RELT"    , "交易关联异常"),
        ("car_risk"         , "RELC"    , "车辆异常"),
        ("comp_risk"        , "ORG"     , "公司异常"),
        ("rule_out"         , "PRDR"    , "准入条件限制")
    ]
    reprs = [(key, f'contains({field}, "{kwd}")', desc)
             for key, kwd, desc in approval_code_cats]
    return reprs


# %%
# TODO
def biztype_cats_reprs(field: str = "biztype"):
    biztype_cats = [
        ("guar_newcar", [100001, 300001], "有担新车"),
        ("guar_usdcar", [200001, 400001], "有担二手车"),
        ("noguar_newcar", [900001, 1100001], "无担新车"),
        ("noguar_usdcar", [800001, 1000001], "无担二手车"),
    ]
    reprs = []
    # 数据总会被转换为数值型
    for key, cats, desc in biztype_cats:
        # cats_cond = '","'.join([str(cat) for cat in cats])
        # reprs.append((key, 'isin(biztype, ["' + cats_cond + '"])', desc))
        cats_cond = ','.join([str(cat) for cat in cats])
        reprs.append((key, 'isin(biztype, [' + cats_cond + '])', desc))
    return reprs


# %%
LOAN_ACC_STATUS = {
    # 0x: 结清、关闭、未激活
    # 1x: 正常
    # 2x: 逾期、催收、追偿
    # 3x: 平仓、止付、冻结、转出
    # 99: 呆账
    "cls": (1           , "正常关闭"),
    "prep": (2          , "提前结清"),
    "repp": (3          , "代偿结清"),
    "nor": (11          , "正常还款"),
    "ovd": (21          , "逾期"),
    "abnor":(31         , "异常"),
    "dum": (99          , "呆账"),
}


def acc_status(field: str = "acc_status"):
    reprs = [("inact", f"{field} < 10", "已关闭"),
             ("nor", f"{field} == 11", "正常"),
             ("ovd", f"({field} > 20) & ({field} < 30)", "逾期"),
             ("abnor", f"({field} > 30) & ({field} < 40)", "异常"),
             ("dum", f"{field} == 99", "呆账"),
             (None, None, None)]
    return reprs


# %%
def merge_certno_perday(df):
    """Merge duplicated records of the same certno per-day.
    """
    import pandas as pd

    def drop_and_merge(sub_df):
        # sub_df = subdf.sort_values("apply_date", ascending=False)
        ser = sub_df.iloc[0]
        app_ret_map = {"accept": 1, "reject": 2, "validation": 3}
        app_ret_map_R = {v: k for k, v in app_ret_map.items()}
        ser["approval_result"] = app_ret_map_R[sub_df["approval_result"]
                                               .map(app_ret_map)
                                               .fillna(3)
                                               .min()]
        ser["approval_codes"] = ",".join(
            set(sub_df["approval_codes"].dropna().values))
        return ser

    df = (df.groupby(["biztype", "certno", "apply_date"])
          .apply(drop_and_merge)
          .reset_index(drop=True))

    return df


TRANS_AUTOFIN_PRETRIAL = {
    "part": "autofin_pretrial",
    "trans": [
        ("apply_doi"        , "day_itvl(apply_date, today)"             , None  , "申请距今日"),
        ("infocode_cats"    , "sep_map(approval_codes, infocode_map)"   , None  , "Infocode类型"),
    ],
    "pre_trans": merge_certno_perday,
}


# %%
TRANS_AUTOFIN_SECTRIAL = {
    "part": "autofin_sectrial",
    "trans": [
        ("apply_doi"        , "day_itvl(apply_date, today)"             , None  , "申请距今日"),
        ("infocode_cats"    , "sep_map(approval_codes, infocode_map)"   , None  , "Infocode类型"),
    ],
    "pre_trans": merge_certno_perday,
}


# %%
TRANS_LOAN_ACC_INFO = {
    "part": "loan_acc_info",
    "trans": [
        ("acc_status"       , "map(acc_status, loan_acc_status)"        , None  , "账户状态"),
    ]
}


# %%
TRANS_LOAN_REPAYMENT_MONTHLY = {
    "part": "loan_repayment_monthly",
    "trans": [
        ("duepay_moi"       , "mon_itvl(duepay_date, today)"            , None  , "应还款距今月"),
        ("repay_moi"        , "mon_itvl(repay_date, today)"             , None  , "实还款距今月"),
    ]
}


# %%
MAPPERS = {
    # "approval_flag"                 : APPROVAL_FLAG,
    "loan_acc_status"               : LOAN_ACC_STATUS,
}

MAPPERS_CODE = {k: {kk: vv[0] for kk, vv in v.items()}
                for k, v in MAPPERS.items()}
MAPPERS_CHN = {k: {kk: vv[1] for kk, vv in v.items()}
               for k, v in MAPPERS.items()}

TRANS_ENV = {
    "map_gov_region_cat"            : map_gov_region_cat,
    "infocode_map"                  : get_infocode_mapper(),
    **MAPPERS_CODE,
}


# %%
TRANS_CONF = {
    TRANS_AUTOFIN_PRETRIAL["part"]          : TRANS_AUTOFIN_PRETRIAL,
    TRANS_AUTOFIN_SECTRIAL["part"]          : TRANS_AUTOFIN_SECTRIAL,
    TRANS_LOAN_REPAYMENT_MONTHLY["part"]    : TRANS_LOAN_REPAYMENT_MONTHLY,
    TRANS_LOAN_ACC_INFO["part"]             : TRANS_LOAN_ACC_INFO,
}


def df_trans_confs():
    trans_conf = []
    for part_name, conf in TRANS_CONF.items():
        rules = [(part_name, key, cond, trans, desc)
                 for key, trans, cond, desc in conf["trans"]]
        trans_conf.extend(rules)
    trans_conf = pd.DataFrame.from_records(
        trans_conf, columns=["part", "key", "cond", "trans", "desc"])

    return trans_conf
