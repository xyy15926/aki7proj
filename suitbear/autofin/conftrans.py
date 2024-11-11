#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: conftrans.py
#   Author: xyy15926
#   Created: 2024-09-23 09:57:58
#   Updated: 2024-11-06 10:23:36
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd
from itertools import product

from suitbear.slp.finer import get_assets_path
from modsbear.locale.govreg import get_chn_govrs


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
def prov_code():
    govrs = get_chn_govrs(1).set_index("id")
    return govrs["name"].to_dict()


# %%
APPR_RES_MAPPER = {
    "accept": (1        , "通过"),
    "reject": (2        , "拒绝"),
    "validation": (3    , "待定"),
}


def appr_res_reprs(field: str = "appr_res"):
    reprs = [
        ("acp", f"{field} == 1", "通过"),
        ("rej", f"{field} == 2", "拒绝"),
        ("vld", f"{field} == 3", "待定"),
        (None, None, None),
    ]
    return reprs


# %%
INFOCODE_MAPPER = {
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


def appr_code_reprs(field: str = "infocode_cats"):
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
    reprs += [(None, None, None)]
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
    for key, cats, desc in biztype_cats:
        # 编码被强制转换为字符串，需添加 `"` 修正类型
        cats_cond = '","'.join([str(cat) for cat in cats])
        reprs.append((key, 'isin(biztype, ["' + cats_cond + '"])', desc))
        # cats_cond = ','.join([str(cat) for cat in cats])
        # reprs.append((key, 'isin(biztype, [' + cats_cond + '])', desc))
    reprs += [(None, None, None)]
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
def merge_certno_perday(df: pd.DataFrame):
    """Merge duplicated records of the same biz with the same certno per-day.
    """
    def drop_and_merge(sub_df: pd.DataFrame):
        # sub_df = subdf.sort_values("apply_date", ascending=False)
        ser = sub_df.iloc[0]
        # ATTENTION: this takes effects just because the precendece of
        # `accept`, `reject` and `validation` are the same with their
        # alphabetic order.
        ser["appr_res"] = sub_df["approval_result"].sort_values().iloc[0]
        ser["appr_codes"] = ",".join(
            set(sub_df["approval_codes"].dropna().values))
        return ser

    df = (df.groupby(["biztype", "channel_code", "certno", "apply_date"])
          .apply(drop_and_merge)
          .reset_index(drop=True))

    return df


TRANS_AUTOFIN_PRETRIAL = {
    "part": "autofin_pretrial",
    "trans": [
        ("cert_prov"        , 'map(certno, prov_code_map, "unknown")'   , None  , "身份证省"),
        ("apply_doi"        , "day_itvl(apply_date, today)"             , None  , "申请距今日"),
        ("appr_doi"         , "day_itvl(approval_date, today)"          , None  , "决策距今日"),
        ("appr_res"         , "map(appr_res, appr_res_mapper)"          , None  , "决策结果"),
        ("infocode_cats"    , "sep_map(appr_codes, infocode_map)"       , None  , "Infocode类型"),
    ],
    # `appr_res` and `appr_codes` are added.
    "pre_trans": merge_certno_perday,
}


# %%
TRANS_AUTOFIN_SECTRIAL = {
    "part": "autofin_sectrial",
    "trans": [
        ("apply_doi"        , "day_itvl(apply_date, today)"             , None  , "申请距今日"),
        ("appr_doi"         , "day_itvl(approval_date, today)"          , None  , "决策距今日"),
        ("appr_res"         , "map(appr_res, appr_res_mapper)"          , None  , "决策结果"),
        ("infocode_cats"    , "sep_map(appr_codes, infocode_map)"       , None  , "Infocode类型"),
    ],
    # `appr_res` and `appr_codes` are added.
    "pre_trans": merge_certno_perday,
}


# %%
TRANS_LOAN_ACC_INFO = {
    "part": "loan_acc_info",
    "trans": [
        ("acc_status"       , "map(acc_status, loan_acc_status)"        , None  , "账户状态"),
        ("acc_start_moi"    , "mon_itvl(loan_date, today)"              , None  , "放款距今月"),
        ("acc_end_moi"      , "mon_itvl(close_date, today)"             , None  , "关闭距今月"),
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
    "appr_res_mapper"               : APPR_RES_MAPPER,
}

MAPPERS_CODE = {k: {kk: vv[0] for kk, vv in v.items()}
                for k, v in MAPPERS.items()}
MAPPERS_CHN = {k: {kk: vv[1] for kk, vv in v.items()}
               for k, v in MAPPERS.items()}

TRANS_ENV = {
    "prov_code_map"                 : prov_code(),
    "infocode_map"                  : INFOCODE_MAPPER,
    **MAPPERS_CODE,
}


# %%
TRANS_CONF = {
    TRANS_AUTOFIN_PRETRIAL["part"]          : TRANS_AUTOFIN_PRETRIAL,
    TRANS_AUTOFIN_SECTRIAL["part"]          : TRANS_AUTOFIN_SECTRIAL,
    TRANS_LOAN_REPAYMENT_MONTHLY["part"]    : TRANS_LOAN_REPAYMENT_MONTHLY,
    TRANS_LOAN_ACC_INFO["part"]             : TRANS_LOAN_ACC_INFO,
}


# %%
def df_trans_confs():
    import pandas as pd

    trans_conf = []
    for part_name, conf in TRANS_CONF.items():
        rules = [(part_name, key, cond, trans, desc)
                 for key, trans, cond, desc in conf["trans"]]
        trans_conf.extend(rules)
    trans_conf = pd.DataFrame.from_records(
        trans_conf, columns=["part", "key", "cond", "trans", "desc"])

    return trans_conf
