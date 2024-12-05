#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: confflat.py
#   Author: xyy15926
#   Created: 2024-09-28 15:24:40
#   Updated: 2024-12-05 17:33:47
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any, TypeVar, TYPE_CHECKING
from collections.abc import Mapping


# %%
AUTOFIN_PRETRIAL = {
    "part": "autofin_pretrial",
    "desc": "预审记录",
    "steps": None,
    "prikey": ["order_no", "certno"],
    "level": 1,
    "fields": [
        # 个体ID信息
        ("order_no"             , None  , "VARCHAR(31)"     , "订单编号"),
        ("certno"               , None  , "VARCHAR(31)"     , "身份证号"),
        ("apply_date"           , None  , "DATE"            , "申请时间"),
        # 关联信息
        ("tel"                  , None  , "VARCHAR(31)"     , "手机号"),
        ("saler_certno"         , None  , "VARCHAR(31)"     , "SP身份证号"),
        # 审批结果
        ("approval_result"      , None  , "VARCHAR(255)"    , "申请结果"),
        ("approval_codes"       , None  , "VARCHAR(255)"    , "申请标签"),
        ("approval_date"        , None  , "DATE"            , "决策时间"),
        # 个体属性信息
        ("biztype"              , None  , "VARCHAR(31)"     , "业务类型"),
        ("age"                  , None  , "INT"             , "年龄"),
        ("channel_code"         , None  , "VARCHAR(31)"     , "渠道编码"),
    ],
}


# %%
AUTOFIN_SECTRIAL = {
    "part": "autofin_sectrial",
    "desc": "资审记录",
    "steps": None,
    "prikey": ["order_no", "certno"],
    "level": 1,
    "fields": [
        # 个体ID信息
        ("order_no"             , None  , "VARCHAR(31)"     , "订单编号"),
        ("apply_date"           , None  , "DATE"            , "申请时间"),
        ("certno"               , None  , "VARCHAR(31)"     , "身份证号"),
        # 关联信息
        ("tel"                  , None  , "VARCHAR(31)"     , "手机号"),
        ("spos_certno"          , None  , "VARCHAR(31)"     , "配偶身份证号"),
        ("spos_tel"             , None  , "VARCHAR(31)"     , "配偶手机号"),
        ("guar1_certno"         , None  , "VARCHAR(31)"     , "担保人身份证号"),
        ("guar1_tel"            , None  , "VARCHAR(31)"     , "担保人手机号"),
        ("guar2_certno"         , None  , "VARCHAR(31)"     , "担保人身份证号"),
        ("guar2_tel"            , None  , "VARCHAR(31)"     , "担保人手机号"),
        ("guar3_certno"         , None  , "VARCHAR(31)"     , "担保人身份证号"),
        ("guar3_tel"            , None  , "VARCHAR(31)"     , "担保人手机号"),
        ("guar4_certno"         , None  , "VARCHAR(31)"     , "担保人身份证号"),
        ("guar4_tel"            , None  , "VARCHAR(31)"     , "担保人手机号"),
        ("guar5_certno"         , None  , "VARCHAR(31)"     , "担保人身份证号"),
        ("guar5_tel"            , None  , "VARCHAR(31)"     , "担保人手机号"),
        ("link1_certno"         , None  , "VARCHAR(31)"     , "联系人身份证号"),
        ("link1_tel"            , None  , "VARCHAR(31)"     , "联系人手机号"),
        ("link2_certno"         , None  , "VARCHAR(31)"     , "联系人身份证号"),
        ("link2_tel"            , None  , "VARCHAR(31)"     , "联系人手机号"),
        ("link3_certno"         , None  , "VARCHAR(31)"     , "联系人身份证号"),
        ("link3_tel"            , None  , "VARCHAR(31)"     , "联系人手机号"),
        ("link4_certno"         , None  , "VARCHAR(31)"     , "联系人身份证号"),
        ("link4_tel"            , None  , "VARCHAR(31)"     , "联系人手机号"),
        ("link5_certno"         , None  , "VARCHAR(31)"     , "联系人身份证号"),
        ("link5_tel"            , None  , "VARCHAR(31)"     , "联系人手机号"),
        ("vin"                  , None  , "VARCHAR(31)"     , "车辆VIN码"),
        ("retailer_orgno"       , None  , "VARCHAR(31)"     , "门店统一社会信用代码"),
        ("retailer_name"        , None  , "VARCHAR(1022)"   , "门店名称"),
        ("res_addr"             , None  , "VARCHAR(1022)"   , "居住地址"),
        ("res_tel"              , None  , "VARCHAR(31)"     , "住址电话"),
        ("domi_addr"            , None  , "VARCHAR(31)"     , "户籍地"),
        ("comp_name"            , None  , "VARCHAR(255)"    , "单位名称"),
        ("comp_addr"            , None  , "VARCHAR(1022)"   , "单位地址"),
        ("comp_tel"             , None  , "VARCHAR(31)"     , "单位联系电话"),
        ("spos_comp_name"       , None  , "VARCHAR(31)"     , "配偶工作单位"),
        ("spos_comp_addr"       , None  , "VARCHAR(31)"     , "配偶工作单位地址"),
        ("spos_comp_tel"        , None  , "VARCHAR(31)"     , "配偶工作单位电话"),
        # 审批结果
        ("approval_result"      , None  , "VARCHAR(255)"    , "申请结果"),
        ("approval_codes"       , None  , "VARCHAR(255)"    , "申请标签"),
        ("approval_date"        , None  , "DATE"            , "决策时间"),
        # 个人信息
        ("maritual_status"      , None  , "VARCHAR(31)"     , "婚姻状态"),
        ("edu_degree"           , None  , "VARCHAR(31)"     , "学历"),
        ("emp_status"           , None  , "VARCHAR(31)"     , "职业"),
        # 累计主体信息
        ("biztype"              , None  , "VARCHAR(31)"     , "业务类型"),
        ("channel_code"         , None  , "VARCHAR(31)"     , "渠道编码"),
    ],
}

# %%
LOAN_ACC_INFO = {
    "part": "loan_acc_info",
    "desc": "借贷账户信息",
    "steps": None,
    "prikey": ["order_no", "certno"],
    "level": 1,
    "fields": [
        ("order_no"             , None  , "VARCHAR(31)"     , "订单编号"),
        ("certno"               , None  , "VARCHAR(31)"     , "身份证号"),
        ("loan_date"            , None  , "DATE"            , "放款时间"),
        ("close_date"           , None  , "DATE"            , "关闭时间"),
        ("debit_card"           , None  , "VARCHAR(31)"     , "划扣卡号"),
        ("acc_status"           , None  , "VARCHAR(31)"     , "账户状态"),
        ("loan_pri"             , None  , "INT"             , "融资金额"),
        ("loan_ppor"            , None  , "FLOAT"           , "融资比例"),
        ("loan_term"            , None  , "INT"             , "期数"),
    ]
}


# %%
LOAN_REPAYMENT_MONTHLY = {
    "part": "loan_repayment_monthly",
    "desc": "还款记录",
    "steps": None,
    "prikey": ["order_no", "certno", "mob"],
    "level": 2,
    "fields": [
        ("order_no"             , None  , "VARCHAR(31)"     , "订单编号"),
        ("certno"               , None  , "VARCHAR(31)"     , "身份证号"),
        ("mob"                  , None  , "INT"             , "账期"),
        # 应还
        ("duepay_amt"           , None  , "FLOAT"           , "应还金额"),
        ("duepay_pri"           , None  , "FLOAT"           , "应还本金"),
        ("duepay_date"          , None  , "FLOAT"           , "应还日期"),
        # 实还
        ("repay_amt"            , None  , "FLOAT"           , "实还金额"),
        ("repay_date"           , None  , "DATE"            , "实还日期"),
        ("repay_card"           , None  , "VARCHAR(31)"     , "还款卡号"),
        # ("repay_debit_owner"    , None  , "VARCHAR(31)"     , "还款银行账户人"),
        # 按日更新逾期情况
        ("ovd_days"             , None  , "INT"             , "逾期天数"),
        ("ovd_amt"              , None  , "INT"             , "逾期金额"),
        ("ovd_pri"              , None  , "INT"             , "逾期本金"),
    ]
}


# %%
AUTOFIN_RETAIL = {
    "part": "autofin_retail",
    "desc": "门店准入",
    "steps": None,
    "prikey": ["order_no", "orgno"],
    "level": 0,
    "fields": [
        ("order_no"             , None  , "VARCHAR(31)"     , "订单号"),
        ("orgno"                , None  , "VARCHAR(31)"     , "统一社会信用代码"),
        ("org_addr"             , None  , "VARCHAR(1022)"   , "门店地址"),
        ("org_name"             , None  , "VARCHAR(255)"    , "门店名称"),
        ("rep_certno"           , None  , "VARCHAR(31)"     , "法人代表身份证号"),
        ("rep_tel"              , None  , "VARCHAR(31)"     , "法人代表手机号"),
        ("apply_date"           , None  , "DATE"            , "申请日期"),
        ("approval_date"        , None  , "DATE"            , "决策日期"),
    ],
}

AUTOFIN_SALER = {
    "part": "autofin_saler",
    "desc": "销售专员准入",
    "steps": None,
    "prikey": ["order_no", "certno"],
    "level": 0,
    "fields": [
        ("order_no"             , None  , "VARCHAR(31)"     , "订单号"),
        ("certno"               , None  , "VARCHAR(31)"     , "SP身份证号"),
        ("tel"                  , None  , "VARCHAR(31)"     , "SP手机号"),
        ("apply_date"           , None  , "DATE"            , "申请时间"),
        ("approval_date"        , None  , "DATE"            , "决策日期"),
    ],
}


# %%
AUTOFIN_PARTS = {
    AUTOFIN_PRETRIAL["part"]: AUTOFIN_PRETRIAL,
    AUTOFIN_SECTRIAL["part"]: AUTOFIN_SECTRIAL,
    LOAN_ACC_INFO["part"]: LOAN_ACC_INFO,
    LOAN_REPAYMENT_MONTHLY["part"]: LOAN_REPAYMENT_MONTHLY,
    AUTOFIN_RETAIL["part"]: AUTOFIN_RETAIL,
    AUTOFIN_SALER["part"]: AUTOFIN_SALER,
}


# %%
def df_flat_confs():
    import pandas as pd

    af_parts = []
    af_fields = []
    for val in AUTOFIN_PARTS.values():
        part_one = {
            "part": val["part"],
            "level": val["level"],
            "prikey": val["prikey"],
            "steps": val["steps"],
            "desc": val["desc"],
        }
        af_parts.append(part_one)
        af_fields.extend([[val["part"], *ele] for ele in val["fields"]])
    af_parts = pd.DataFrame(af_parts)
    af_fields = pd.DataFrame.from_records(
        af_fields, columns=["part", "field", "step", "dtype", "desc"])

    return af_parts, af_fields
