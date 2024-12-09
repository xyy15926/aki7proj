#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: conftrans.py
#   Author: xyy15926
#   Created: 2024-09-23 09:57:58
#   Updated: 2024-12-09 19:29:59
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING
from itertools import product

import pandas as pd
from flagbear.slp.finer import get_assets_path
from modsbear.locale.govreg import get_chn_govrs

INFOCODE_MAPPER_FILE = get_assets_path() / "autofin/infocode_mapper.xlsx"


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
    return govrs[["PinYin", "name"]].T.to_dict("list")


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
# 4 位分类编码
# 1. 1st：主体
#   A - 承租人
#   B - 配偶
#   C - 担保人
#   D - 联系人
#   E - 车辆
#   F - 金融中介
#   G - 交易门店
#   H - 工作单位
#   M - 企业申请主体
#   Z - 其他
# 2. 2nd：维度
#   Application - 信贷申请记录
#   aCcount - 信贷账户记录
#   Overdue - 逾期记录
#   Publication - 公共事件
#   Indivisual - 个人信息
#   Job - 职业、单位
#   aDdress - 地址
#   tElephone - 电话
#   Relation - 主体关联关系（配合身份证号唯一第一等节点关联图谱）
#   Score - 综合评分
#   Tag - 其他标签
# 3. 3rd: 数据来源
#   X - 行内自有数据
#   Y - 外部三方数据
#   Z - 人行征信数据
# 4. 4th: 位序
APPR_CODE_CATS = {
    "AA": (
        "AA",
        "承租人信贷申请记录",
        [
            ("AAXA", "历史申请记录_行内历史"),
            ("AAXB", "历史申请记录_行内2年内"),
            ("AAXC", "历史申请记录_行内1年内"),
            ("AAXD", "历史申请记录_行内6月内"),
            ("AAXE", "历史申请记录_行内3月内"),
            ("AAXF", "历史申请记录_行内1月内"),
            ("AAXG", "在途订单_行内当前在途"),
            ("AAYA", "多头申请_三方历史"),
            ("AAYB", "多头申请_三方2年内"),
            ("AAYC", "多头申请_三方1年内"),
            ("AAYD", "多头申请_三方6月内"),
            ("AAYE", "多头申请_三方3月内"),
            ("AAYF", "多头申请_三方1月内"),
            ("AAZA", "多头申请_人行历史"),
            ("AAZB", "多头申请_人行2年内"),
            ("AAZC", "多头申请_人行1年内"),
            ("AAZD", "多头申请_人行6月内"),
            ("AAZE", "多头申请_人行3月内"),
            ("AAZF", "多头申请_人行1月内"),
        ],
    ),
    "AC": (
        "AC",
        "承租人信贷账户情况",
        [
            ("ACXA", "账户数量_行内"),
            ("ACYA", "账户数量_三方"),
            ("ACZA", "账户数量_人行"),
            ("ACZB", "负债水平_人行"),
            ("ACZC", "信贷记录短_人行白户"),
        ],
    ),
    "AD": (
        "AD",
        "承租人相关地址特征",
        [
            ("ADXA", "申请人居住地限制"),
            ("ADXB", "申请人身份证地限制"),
            ("ADXC", "申请人地址关联异常_行内"),
        ],
    ),
    "AE": (
        "AE",
        "承租人手机号特征",
        [
            ("AEXA", "手机号非本人所有"),
            ("AEXB", "手机号非三大运营商"),
            ("AEXC", "手机号在网时长短"),
            ("AEXD", "手机号在网状态异常"),
            ("AEXE", "手机号关联身份证异常_行内"),
            ("AEYA", "手机号关联身份证异常_三方"),
        ],
    ),
    "AI": (
        "AI",
        "申请人个人信息",
        [
            ("AIXA", "申请人年龄限制"),
            ("AIXB", "申请人证件过期"),
            ("AIXC", "申请人婚姻状态异常_行内"),
        ],
    ),
    "AJ": (
        "AJ",
        "申请人职业单位信息",
        [
            ("AJXA", "申请人职业限制_行内"),
            ("AJZA", "申请人职业限制_人行"),
            ("AJYA", "申请人职业限制_三方"),
        ],
    ),
    "AO": (
        "AO",
        "申请人历史逾期信息",
        [
            ("AOXA", "行内信贷逾期"),
            ("AOYA", "三方信贷逾期"),
            ("AOZA", "人行逾期_历史"),
            ("AOZB", "人行逾期_2年内"),
            ("AOZC", "人行逾期_1年内"),
            ("AOZD", "人行逾期_6月内"),
            ("AOZE", "人行逾期_3月内"),
            ("AOZF", "人行逾期_当前"),
            ("AOZG", "人行类逾期记录"),
        ],
    ),
    "AP": (
        "AP",
        "申请人官方公开信息",
        [
            ("APYA", "法院被执行记录_三方"),
            ("APYB", "法院判决记录_三方"),
            ("APYC", "公安执法记录_三方"),
            ("APYD", "驾照状态异常_三方"),
            ("APYE", "公安失联名单_三方"),
            ("APZA", "法院被执行记录_人行"),
            ("APZB", "法院判决记录_人行"),
            ("APZC", "行政处罚记录_人行"),
            ("APZD", "非信贷欠费_人行"),
            ("APZE", "低保救助_人行"),
        ],
    ),
    "AR": (
        "AR",
        "申请人关联关系特征",
        [
            ("ARXA", "关联手机号异常_行内"),
            ("ARXB", "关联配偶异常_行内"),
            ("ARXC", "关联工作单位异常_行内"),
            ("ARXD", "关联车辆异常_行内"),
            ("ARYA", "关联手机号异常_三方"),
            ("ARYB", "关联车辆异常_三方"),
            ("ARYC", "关联高危人员_三方"),
            ("ARZA", "关联手机号异常_人行"),
        ],
    ),
    "AS": (
        "AS",
        "综合评分",
        [
            ("ASXA", "评分异常_内部"),
            ("ASYA", "评分异常_三方"),
        ],
    ),
    "AT": (
        "AT",
        "申请人风险标签",
        [
            ("ATYA", "欺诈标签_三方"),
            ("ATYB", "仿冒伪造标签_三方"),
            ("ATYZ", "风险标签_三方"),
            ("ATYC", "套现标签_三方"),
            ("ATYD", "车交易记录异常_三方"),
        ],
    ),
    "BZ": (
        "BZ",
        "配偶异常",
        [
            ("BAXA", "配偶历史申请行为_行内"),
            ("BEXE", "配偶手机号关联信息异常_行内"),
            ("BPYA", "配偶法院被执行记录_三方"),
            ("BEXZ", "配偶关联网络异常_行内"),
            ("BJXA", "配偶职业限制_行内"),
        ],
    ),
    "CZ": (
        "CZ",
        "担保人异常",
        [
            ("CAXA", "担保人历史申请行为_行内"),
            ("CEXE", "担保人手机号关联信息异常_行内"),
            ("CPYA", "配偶法院被执行记录_三方"),
            ("CEXZ", "担保人关联网络异常_行内"),
            ("CJXA", "担保人从业限制_行内"),
        ],
    ),
    "DZ": (
        "DZ",
        "联系人异常",
        [
            ("DAXA", "联系人历史申请行为_行内"),
            ("DEXE", "联系人手机号关联信息异常_行内"),
            ("DPYA", "联系人法院被执行记录_三方"),
            ("DEXZ", "联系人关联网络异常_行内"),
            ("DJXA", "联系人从业限制_行内"),
        ],
    ),
    "EZ": (
        "EZ",
        "车辆异常",
        [
            ("EIXA", "车辆融资比例异常"),
            ("EIXB", "车辆融资额异常"),
            ("EIXC", "车型限制"),
            ("EIXD", "车辆品牌限制"),
            ("EIXE", "车辆行驶里程限制"),
            ("EIYA", "车辆评估价过高"),
            ("ERXA", "车辆关联异常_行内"),
            ("ERYA", "车辆关联欺诈业务、机构、人群_三方"),
            ("ETYA", "车辆权属状态异常"),
            ("ETYB", "车况异常"),
            ("ETYC", "车辆维保记录异常"),
            ("ETYD", "车辆交易过户记录、数量异常"),
        ],
    ),
    "GZ": (
        "GZ",
        "申请门店异常",
        [
            ("GRXA", "申请门店关联异常"),
        ],
    ),
    "HZ": (
        "HZ",
        "申请人单位异常",
        [
            ("HRXA", "工作单位关联异常"),
            ("HEXA", "工作单位电话关联异常"),
        ],
    ),
    "MI": (
        "MI",
        "企业申请主体申请异常",
        [
            ("MIXA", "企业补充记录异常"),
            ("MIYB", "企业注册信息异常"),
            ("MIYC", "企业经营状态异常"),
            ("MIYD", "企业法人代表异常"),
        ],
    ),
    "OTH": (
        "OTH",
        "其他原因",
        [
            ("OTHA", "业务因素异常"),
            ("OTHC", "业务白名单"),
            ("OTHB", "非业务因素异常"),
            ("OTHD", "业务黑名单"),
        ],
    ),
}


def gen_appr_code_mapper_lv21() -> dict:
    imdf = pd.read_excel(INFOCODE_MAPPER_FILE)
    immap = {}
    for code, cc, cd in imdf[["code", "code_cat", "cat_desc"]].values:
        if pd.notna(cc):
            immap[code] = (cc, cd)
    return immap


def gen_appr_code_mapper_lv10() -> dict:
    immap = {}
    for part, (klv0, dlv0, items) in APPR_CODE_CATS.items():
        for klv1, dlv1 in items:
            immap[klv1] = (klv0, dlv0)
    return immap


def appr_catlv1_reprs(field: str = "appr_catlv1"):
    reprs = []
    cats = [
        "ACZC",
        "AEXA",
        "AEXB",
        "AEXC",
        "AEXD",
    ]
    for part, (klv0, dlv0, items) in APPR_CODE_CATS.items():
        for klv1, dlv1 in items:
            if klv1 in cats:
                reprs.append((klv1, f'contains({field}, "{klv1}")', dlv1))
    return reprs


def appr_catlv0_reprs(field: str = "appr_catlv0"):
    reprs = []
    for part, (klv0, dlv0, items) in APPR_CODE_CATS.items():
        reprs.append((klv0, f'contains({field}, "{klv0}")', dlv0))
    return reprs


# %%
def biztype_cats_reprs(field: str = "biztype"):
    biztype_cats = [
        ("gncar", [1, 3, 100001, 300001, 300002,], "有担新车"),
        ("gscar", [2, 4, 200001, 200003, 400001, 400002, 400003], "有担二手车"),
        ("ngncar", [7, 9, 700001, 700002, 900001, 900002], "无担新车"),
        ("ngscar", [8, 10, 800001, 800002, 1000001, 1000002], "无担二手车"),
        ("gagrm", [1500001, 1600001], "农机"),
        ("gmcar", [1800001, 1900001], "出行融"),
        ("inloan", [5, 6, 11, 12, 500001, 600001, 1000001, 1200001], "租中监控"),
        ("othbiz", [30, 40, 99, 3000001, 4000001, 9900001, 1300001, 1400001, 1500001],
         "其他事件"),
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
    "desc": "预审记录",
    "level": 1,
    "prikey": ["order_no", "certno"],
    "from_": ["autofin_pretrial"],
    "joinkey": None,
    "trans": [
        ("cert_prov"        , 'map(certno, prov_code_map, "unknown")'   , None  , "身份证省"),
        ("apply_doi"        , "day_itvl(apply_date, today)"             , None  , "申请距今日"),
        ("appr_doi"         , "day_itvl(approval_date, today)"          , None  , "决策距今日"),
        ("appr_res"         , "map(appr_res, appr_res_mapper)"          , None  , "决策结果"),
        ("appr_catlv1"      , "sep_map(appr_codes, appr_cats_mapper_lv21)"      , None  , "标签分类LV1"),
        ("appr_catlv0"      , "list_map(appr_catlv1, appr_cats_mapper_lv10)"    , None  , "标签分类LV0"),
    ],
    # `appr_res` and `appr_codes` are added.
    "pre_trans": merge_certno_perday,
}


# %%
TRANS_AUTOFIN_SECTRIAL = {
    "part": "autofin_sectrial",
    "desc": "资审记录",
    "level": 1,
    "prikey": ["order_no", "certno"],
    "from_": ["autofin_sectrial"],
    "joinkey": None,
    "trans": [
        ("apply_doi"        , "day_itvl(apply_date, today)"             , None  , "申请距今日"),
        ("appr_doi"         , "day_itvl(approval_date, today)"          , None  , "决策距今日"),
        ("appr_res"         , "map(appr_res, appr_res_mapper)"          , None  , "决策结果"),
        ("appr_catlv1"      , "sep_map(appr_codes, appr_cats_mapper_lv21)"      , None  , "标签分类LV1"),
        ("appr_catlv0"      , "list_map(appr_catlv1, appr_cats_mapper_lv10)"    , None  , "标签分率LV0"),
    ],
    # `appr_res` and `appr_codes` are added.
    "pre_trans": merge_certno_perday,
}


# %%
TRANS_LOAN_ACC_INFO = {
    "part": "loan_acc_info",
    "desc": "借贷账户信息",
    "level": 1,
    "prikey": ["order_no", "certno"],
    "from_": ["loan_acc_info"],
    "joinkey": None,
    "trans": [
        ("acc_status"       , "map(acc_status, loan_acc_status)"        , None  , "账户状态"),
        ("acc_start_moi"    , "mon_itvl(loan_date, today)"              , None  , "放款距今月"),
        ("acc_end_moi"      , "mon_itvl(close_date, today)"             , None  , "关闭距今月"),
    ]
}


# %%
TRANS_LOAN_REPAYMENT_MONTHLY = {
    "part": "loan_repayment_monthly",
    "desc": "还款记录",
    "level": 2,
    "prikey": ["order_no", "certno", "mob"],
    "from_": ["loan_repayment_monthly"],
    "joinkey": None,
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
    "prov_code_map"                 : prov_code(),
    "appr_cats_mapper_lv21"         : gen_appr_code_mapper_lv21(),
    "appr_cats_mapper_lv10"         : gen_appr_code_mapper_lv10(),
}

MAPPERS_CODE = {k: {kk: vv[0] for kk, vv in v.items()}
                for k, v in MAPPERS.items()}
MAPPERS_CHN = {k: {kk: vv[1] for kk, vv in v.items()}
               for k, v in MAPPERS.items()}


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
