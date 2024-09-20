#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: conftrans.py
#   Author: xyy15926
#   Created: 2024-09-08 20:59:28
#   Updated: 2024-09-20 14:39:35
#   Description:
# ---------------------------------------------------------

# %%
REPAY_STATUS = {
    "1": (1  , "逾期1-30天"),
    "2": (2  , "逾期31-60天"),
    "3": (3  , "逾期61-90天"),
    "4": (4  , "逾期91-120天"),
    "5": (5  , "逾期121-150天"),
    "6": (6  , "逾期151-180天"),
    "7": (7  , "逾期180天以上"),
    "*": (0  , "当月不需要还款且之前没有拖欠"),
    "#": (0  , "未知"),
    "/": (0  , "跳过"),
    "A": (0  , "账单日调整,当月不出单"),
    "B": (7  , "呆账"),
    "C": (0  , "结清、正常销户"),
    "D": (3  , "担保人代还"),
    "G": (7  , "（非正常销户）结束"),
    "M": (0  , "约定还款日后月底前还款"),
    "N": (0  , "正常还款"),
    "Z": (3  , "以资抵债"),
}
REPAY_STATUS_SPEC = {
    "1": (1  , "逾期1-30天"),
    "2": (2  , "逾期31-60天"),
    "3": (3  , "逾期61-90天"),
    "4": (4  , "逾期91-120天"),
    "5": (5  , "逾期121-150天"),
    "6": (6  , "逾期151-180天"),
    "7": (7  , "逾期180天以上"),
    "*": (-1 , "当月不需要还款且之前没有拖欠"),
    "#": (-1 , "未知"),
    "/": (-1 , "跳过"),
    "A": (-1 , "账单日调整,当月不出单"),
    "B": (71 , "呆账"),
    "C": (-1 , "结清、正常销户"),
    "D": (31 , "担保人代还"),
    "G": (72 , "（非正常销户）结束"),
    "M": (-1 , "约定还款日后月底前还款"),
    "N": (0  , "正常还款"),
    "Z": (32 , "以资抵债"),
}
PBOC_ACC_REPAY_60_MONTHLY = {
    "part": "pboc_acc_repay_60_monthly",
    "trans": [
        ["acc_repay_status"     , "map(PD01ED01, repay_status)"                 , None  , "还款状态"],
        ["acc_repay_status_spec", "map(PD01ED01, repay_status_spec)"            , None  , "还款状态"],
        ["acc_repay_status8"    , "cb_max(cb_min(acc_repay_status_spec, 8), 0)" , None  , "还款状态"],
        ["acc_repay_moi"        , "mon_itvl(PD01ER03, today)"                   , None  , "还款月距今月"]
    ],
}


def repay_status(field: str = "acc_repay_status"):
    # repay_status = [0, 1, 2, 3, 4, 5, 6, 7]
    reprs = []
    # Attention: 已经上线了，不能修改 `le`，好烦:(
    reprs += [(f"le{rs}", f"{field} >= {rs}", f"还款状态大于等于{rs}")
              for rs in [1, 3]]
    reprs += [(f"eq{rs}", f"{field} == {rs}", f"还款状态为{rs}")
              for rs in [2, 7]]
    reprs += [(None, None, None)]
    return reprs


def repay_status_spec(field: str = "acc_repay_status_spec"):
    reprs = [
        ("gpaid"    , f"{field} == 31"  , "担保人代还"),
        ("asset"    , f"{field} == 32"  , "以资抵债"),
        ("dum"      , f"{field} == 71"  , "呆账"),
        ("abcls"    , f"{field} == 72"  , "（非正常销户）结束"),
        ("npaid"    , f"{field} == 0"   , "正常还款"),
        ("skip"     , f"{field} == -1"  , "跳过还款"),
    ]
    reprs += [(None, None, None)]
    return reprs


#  %%
SPECIAL_TRANS_TYPE = {
    # 性质逐渐恶劣
    "1": (11        , "展期"),
    "2": (31        , "担保人(第三方)代偿"),
    "3": (32        , "以资抵债"),
    "4": (1         , "提前还款"),
    "5": (2         , "提前结清"),
    "6": (33        , "强制平仓,未结清"),
    "7": (25        , "强制平仓,已结清"),
    "8": (34        , "司法追偿"),
    "9": (14        , "其他"),
    "11": (21       , "债务减免"),
    "12": (22       , "资产剥离"),
    "13": (23       , "资产转让"),
    "14": (12       , "信用卡个性化分期"),
    "16": (13       , "落实金融纾困等政策银行主动延期"),
    "17": (24       , "强制平仓"),
}
PBOC_ACC_SPECIAL_TRANS = {
    "part": "pboc_acc_special_trans",
    "trans": [
        ["acc_special_trans_type", "map(PD01FD01, special_trans_type)",
         None  , "特殊交易类型"],
        ["acc_special_trans_moi", "mon_itvl(PD01FR01, today)",
         None, "特殊交易距今月"],
    ],
}


def special_trans(field: str = "acc_special_trans_type"):
    reprs = [
        ("early", f"{field} < 10", "提前还款"),
        ("exhib", f"({field} > 10) & ({field} < 20)", "展期"),
        ("asset", f"({field} > 20) & ({field} < 30)",
         "资产剥离、转让、债务减免、平仓"),
        ("ovd", f"{field} > 30", "代偿、以资抵债"),
    ]
    return reprs


# %%
SPECIAL_ACCD_TYPE = {
    # R2 账户
    "11":(11        , "信用卡因调整账单日本月不出单"),
    "12":(12        , "已注销信用卡账户重启"),
    # D1 账户，“产品说明” 中缺失此编码，但是有说 “特殊事件” 段对 D1 账户段存在
    "21":(21        , "转出（指受托管理的贷款，因委托人变更，原受托人无法继续报送数据）"),
}
PBOC_ACC_SPECIAL_ACCD = {
    "part": "pboc_acc_special_accd",
    "trans": [
        ["acc_special_accd_type", "map(PD01GD01, special_accd_type)",
         None, "特殊事件类型"],
        ["acc_special_accd_moi", "mon_itvl(PD01GR01, today)",
         None, "特殊事件距今月"],
    ],
}


def special_accd(field: str = "acc_special_accd_type"):
    reprs = [
        ("nor", f"{field} < 20", "信用卡调整"),
        ("ovd", f"{field} > 20", "转出")
    ]
    return reprs


# %%
PBOC_ACC_SPECIAL_INSTS = {
    "part": "pboc_acc_special_insts",
    "trans": [
        ["acc_special_insts_moi_start", "mon_itvl(PD01HR01, today)",
         None, "大额分期起始距今月"],
        ["acc_special_insts_moi_end", "mon_itvl(PD01HR02, today)",
         None, "大额分期结束距今月"],
        # TODO 字段加工逻辑有问题
        ["acc_special_insts_monthly_repay",
         "sdiv(PD01HJ02, cb_max(mon_itvl(PD01HR02, PD01HR01), 1))",
         None, "大额分期月均还款"],
    ],
}


# %%
CDR_CAT = {
    "D1": (1        , "非循环贷账户"),
    "R1": (3        , "循环贷账户"),
    "R2": (4        , "贷记卡账户"),
    "R3": (5        , "准贷记卡账户"),
    "R4": (2        , "循环额度下分账户"),
    "C1": (99       , "催收账户"),
}


def acc_cat(field: str = "acc_cat", choices: list = None):
    acc_cats = {
        "c1": ("c1", f"{field} == 99", "c1"),
        "d1": ("d1", f"{field} == 1", "d1"),
        "r4": ("r4", f"{field} == 2", "r4"),
        "r1": ("r1", f"{field} == 3", "r1"),
        "r2": ("r2", f"{field} == 4", "r2"),
        "r2cny": ("r2cny", f"({field} == 4) & (PD01AD04 == \"CNY\")", "r2cny"),
        "r3": ("r3", f"{field} == 5", "r3"),
        "d1r4": ("d1r4", f"{field} <= 2", "d1r4"),
        "d1r41": ("d1r41", f"{field} <= 3", "d1r41"),
        "r23": ("r23", f"({field} >= 4) & ({field} <= 5)", "r23"),
        "r281": ("r281", f"({field} == 4) & (acc_biz_cat == 221)", "r281"),
        "r282": ("r282", f"({field} == 4) & (acc_biz_cat == 231)", "r282"),
        # 有些机构会直接报普通贷记卡 `81`、授信额度为 0 做大额专项分期，而不是直接报 `82`
        "r2spec": ("r2spec",
                   f"(({field} == 4) & (acc_biz_cat == 231)) | "
                   f"((acc_biz_cat == 221) & (PD01AJ02 == 0))",
                   "r2spec"),
    }
    if choices is None:
        return list(acc_cats.values()) + [(None, None, None)]
    else:
        return [acc_cats[k] for k in choices] + [(None, None, None)]


# %%
BIZ_CAT = {
    # 贷款类业务
    "11": (111      , "个人住房商业贷款"),
    "12": (112      , "个人商用房(含商住两用)贷款"),
    "13": (113      , "个人住房公积金贷款"),
    "21": (121      , "个人汽车消费贷款"),
    "31": (131      , "个人助学贷款"),
    "32": (132      , "国家助学贷款"),
    "33": (133      , "商业助学贷款"),
    "41": (141      , "个人经营性贷款"),
    "42": (142      , "个人创业担保贷款"),
    "51": (143      , "农户贷款"),
    "52": (144      , "经营性农户贷款"),
    "53": (151      , "消费性农户贷款"),
    "91": (152      , "其他个人消费贷款"),
    "99": (161      , "其他贷款"),
    # 信用卡类业务
    "71": (211      , "准贷记卡"),
    "81": (221      , "贷记卡"),
    "82": (231      , "大额专项分期卡"),
    # 证券融资类业务
    "61": (311      , "约定购回式证券交易"),
    "62": (321      , "股票质押式回购交易"),
    "63": (331      , "融资融券业务"),
    "64": (341      , "其他证券类融资"),
    # 融资租赁业务
    "92": (411      , "融资租赁业务"),
    # 资产处置
    "A1": (511      , "资产处置"),
    # 代偿
    "B1": (611      , "代偿债务"),
}


def biz_cat(field: str):
    biz_cat = [
        ("biz_cat_loan", f"{field} < 200", "贷款业务"),
        ("biz_cat_card", f"({field} > 200) & ({field} < 300)", "信用卡业务"),
        ("biz_cat_housing", f"{field} < 120", "房贷业务"),
        ("biz_cat_auto", f"({field} > 120) & ({field} < 130)", "车贷业务"),
        ("biz_cat_edu", f"({field} > 130) & ({field} < 140)", "教育贷款业务"),
        ("biz_cat_biz", f"({field} > 140) & ({field} < 150)", "经营贷、农业贷业务"),
        ("biz_cat_comsu", f"({field} > 150) & ({field} < 160)", "消费贷业务"),
        ("biz_cat_security", f"({field} > 300) & ({field} < 400)", "证券融资业务"),
        ("biz_cat_leasing", f"({field} > 400) & ({field} < 500)", "融资租赁业务"),
        ("biz_cat_dum", f"{field} > 500", "资产处置、垫款业务"),
        (None, None, None),
    ]
    return biz_cat


# %%
ORG_CAT = {
    # 银行
    "11": (11       , "商业银行"),
    "12": (12       , "村镇银行"),
    "14": (14       , "住房储蓄银行"),
    "15": (15       , "外资银行"),
    # 非银金融机构
    "16": (27       , "财务公司"),
    "21": (21       , "信托公司"),
    "22": (22       , "融资租赁公司"),
    "23": (23       , "汽车金融公司"),
    "24": (24       , "消费金融公司"),
    "25": (25       , "贷款公司"),
    "26": (26       , "金融资产管理公司"),
    "31": (31       , "证券公司"),
    "41": (41       , "保险公司"),
    "51": (51       , "小额贷款公司"),
    "53": (53       , "融资担保公司"),
    "54": (54       , "保理公司"),
    # 其他机构
    "52": (98       , "公积金管理中心"),
    "99": (99       , "其他机构"),
}


def org_cat(field: str):
    orgs = [
        ("org_allbank", f"{field} < 20", "银行机构"),
        ("org_nbank", f"({field} > 20) & ({field} < 60)", "非银机构"),
        ("org_other", f"{field} > 90", "其他机构"),
    ]
    orgs += [
        ("dorg_combank", f"{field} == 11", "商业银行"),
        ("dorg_conbank", f"{field} == 12", "村镇银行"),
        ("dorg_leasing", f"{field} == 22", "融资租赁"),
        ("dorg_autofin", f"{field} == 23", "汽金公司"),
        ("dorg_comsufin", f"{field} == 24", "消金公司"),
        ("dorg_loan", f"{field} == 25", "贷款公司"),
        ("dorg_security", f"{field} == 31", "证券公司"),
        ("dorg_insur", f"{field} == 41", "保险"),
        ("dorg_sloan", f"{field} == 51", "小额贷款公司"),
        ("dorg_guar", f"{field} == 53", "融担公司"),
    ]
    return orgs + [(None, None, None),]


# %%
# 0x: 结清、关闭、未激活
# 1x: 正常
# 2x: 逾期、催收、追偿
# 3x: 平仓、止付、冻结、转出
# 99: 呆账
C1_ACC_STATUS = {
    "1": (22        , "催收"),
    "2": (0         , "结束"),
}
D1_ACC_STATUS = {
    "1": (11        , "正常"),
    "2": (21        , "逾期"),
    "3": (0         , "结清"),
    "4": (99        , "呆账"),
    "5": (2         , "转出"),
    "6": (31        , "担保物不足"),
    "7": (32        , "强制平仓"),
    "8": (23        , "司法追偿"),
}
R4_ACC_STATUS = {
    "1": (11        , "正常"),
    "2": (21        , "逾期"),
    "3": (0         , "结清"),
    "4": (99        , "呆账"),
    "6": (31        , "担保物不足"),
    "8": (23        , "司法追偿"),
}
R1_ACC_STATUS = {
    "1": (11        , "正常"),
    "2": (21        , "逾期"),
    "3": (0         , "结清"),
    "4": (99        , "呆账"),
    "5": (33        , "银行止付"),
    "6": (31        , "担保物不足"),
    "8": (23        , "司法追偿"),
}
R23_ACC_STATUS = {
    "1": (11        , "正常"),
    "2": (35        , "冻结"),
    "3": (34        , "止付"),
    "31": (33        , "银行止付"),
    "4": (0         , "销户"),
    "5": (99        , "呆账"),
    "6": (1         , "未激活"),
    "8": (23        , "司法追偿"),
}


def acc_status(field: str = "mixed_acc_status"):
    acc_status = [
        ("inact", f"{field} < 10", "关闭、未激活、转出"),
        ("nor", f"{field} == 11", "正常"),
        ("ovd", f"({field} > 20) & ({field} < 30)", "逾期"),
        ("abnor", f"({field} > 30) & ({field} < 40)", "异常"),
        ("dum", f"{field} == 99", "呆账"),
    ]
    acc_status += [
        ("nocls", f"{field} > 0", "未关闭"),
        ("active", f"{field} > 10", "活跃"),
        ("updt", f"({field} > 10) & ({field} < 40)", "持续更新"),
    ]
    return acc_status + [(None, None, None),]


# %%
LVL5_STATUS = {
    "1": (1         , "正常"),
    "2": (2         , "关注"),
    "3": (3         , "次级"),
    "4": (4         , "可疑"),
    "5": (5         , "损失"),
    "6": (6         , "违约"),
    "9": (1         , "未分类"),
}


def lvl5_status(field: str):
    lvl5_status = [
        ("lvl5_nor", f"{field} == 1", "五级分类正常"),
        ("lvl5_con", f"{field} == 2", "五级分类关注"),
        ("lvl5_ovd", f"{field} >= 2", "五级分类逾期"),
        ("lvl5_inf", f"{field} >= 3", "五级分类次级及以上"),
        (None, None, None),
    ]
    return lvl5_status


# %%
TRANS_STATUS = {
    "0": (0         , "债务人即将违约时自动垫款"),
    "1": (1         , "逾期1-30天"),
    "2": (2         , "逾期31-60天"),
    "3": (3         , "逾期61-90天"),
    "4": (4         , "逾期91-120天"),
    "5": (5         , "逾期121-150天"),
    "6": (6         , "逾期151-180天"),
    "7": (7         , "逾期180天以上"),
    "9": (0         , "未知"),
}


def trans_status(field: str = "acc_trans_status"):
    trans_status = [(f"trans_status_eq{ts}", f"{field}== {ts}",
                     f"转移时状态为{ts}")
                    for ts in [0, 1, 2, 3, 4, 5, 6, 7]]
    return trans_status


# %%
REPAY_TYPE = {
    "11": (11       , "分期等额本息"),
    "12": (12       , "分期等额本金"),
    "13": (13       , "到期还本分期结息"),
    "14": (14       , "等比累进分期还款"),
    "15": (15       , "等额累进分期还款"),
    "19": (19       , "其他类型分期还款"),
    "21": (21       , "到期一次还本付息"),
    "22": (22       , "预先付息到期还本"),
    "23": (23       , "随时还"),
    "29": (29       , "其他"),
    "31": (31       , "按期结息，到期还本"),
    "32": (32       , "按期结息，自由还本"),
    "33": (33       , "按期计算还本付息"),
    "39": (39       , "循环贷款下其他还款方式"),
    "90": (90       , "不区分还款方式"),
}
REPAY_FREQ = {
    "01": (1        , "日"),
    "02": (2        , "周"),
    "03": (3        , "月"),
    "04": (4        , "季"),
    "05": (5        , "半年"),
    "06": (6        , "年"),
    "07": (7        , "一次性"),
    "08": (8        , "不定期"),
    "12": (12       , "旬"),
    "13": (13       , "双周"),
    "14": (14       , "双月"),
    "99": (99       , "其他"),
}


# %%
GUAR_TYPE = {
    # 1: 无担保
    # 1x: 物抵质押
    # 2x: 带人保证
    # 3x: 其他担保
    # 99: 其他
    "1": (11        , "质押"),
    "2": (12        , "抵押"),
    "3": (21        , "保证"),
    "4": (1         , "信用/无担保"),
    "5": (22        , "组合（含保证）"),
    "6": (13        , "组合（不含保证）"),
    "7": (31        , "农户联保"),
    "8": (99        , "其他"),
}


def guar_type(field: str = "acc_guar_type"):
    reprs = [
        ("no_guar", f"{field} < 10", "信用无担保"),
        ("with_gtor", f"({field} > 10) & ({field} < 20)", "带保证人担保"),
        ("with_gdge", f"({field} > 20) & ({field} < 30)", "抵质押担保"),
        ("other_guar", f"{field} == 99", "其他担保方式"),
        (None, None, None),
    ]
    return reprs


# %%
PBOC_ACC_INFO_BASIC = [
    # 账户信息
    ["acc_cat"                      , "map(PD01AD01, cdr_cat)"                      , None  , "账户类型"],
    ["acc_org_cat"                  , "map(PD01AD02, org_cat)"                      , None  , "管理机构类型"],
    ["acc_biz_cat"                  , "map(PD01AD03, biz_cat)"                      , None  , "账户业务类型"],
    ["acc_repay_freq"               , "map(PD01AD06, repay_freq)"                   , None  , "D1R41账户还款频率"],
    ["acc_trans_status"             , "map(PD01AD10, trans_status)"                 , None  , "C1账户转移时状态"],
    ["acc_moi_range"                , "mon_itvl(PD01AR02, PD01AR01)"                , None  , "账户预期月数"],
    ["acc_moi_start"                , "mon_itvl(PD01AR01, today)"                   , None  , "账户起始距今月"],
    ["acc_doi_start"                , "day_itvl(PD01AR01, today)"                   , None  , "账户起始距今日"],
    ["acc_moi_end"                  , "mon_itvl(PD01AR02, today)"                   , None  , "账户（预期）结束距今月"],    # Mixed: mixed_acc_moi_folw
    ["acc_guar_type"                , "map(PD01AD07, guar_type)"                    , None  , "账户担保方式"],
    ["acc_lmt"                      , "cb_fst(cb_fst(PD01AJ01, PD01AJ02), PD01AJ03)", None  , "账户借款、授信额"],
]
PBOC_ACC_INFO_LATEST = [
    # 最新表现
    ["cur_acc_status"       , "map(PD01BD01, c1_acc_status)"    , "acc_cat == 99"                   , "最近状态"],          # Mixed: mixed_acc_status
    ["cur_acc_status"       , "map(PD01BD01, d1_acc_status)"    , "acc_cat == 1"                    , "最近状态"],          # Mixed: mixed_acc_status
    ["cur_acc_status"       , "map(PD01BD01, r4_acc_status)"    , "acc_cat == 2"                    , "最近状态"],          # Mixed: mixed_acc_status
    ["cur_acc_status"       , "map(PD01BD01, r1_acc_status)"    , "acc_cat == 3"                    , "最近状态"],          # Mixed: mixed_acc_status
    ["cur_acc_status"       , "map(PD01BD01, r23_acc_status)"   , "(acc_cat >= 4) & (acc_cat <= 5)" , "最近状态"],          # Mixed: mixed_acc_status
    ["cur_lvl5_status"      , "map(PD01BD03, lvl5_status)"      , None                              , "最近5级分类"],       # Mixed: mixed_lvl5_status
    ["cur_repay_status"     , "map(PD01BD04, repay_status)"     , None                              , "最近还款状态"],
    ["cur_moi_closed"       , "mon_itvl(PD01BR01, today)"       , None                              , "最近关闭时间"],      # Mixed: mixed_acc_moi_folw
    ["cur_doi_last_repay"   , "day_itvl(PD01BR02, today)"       , None                              , "最近还款距今日"],    # Mixed: mixed_doi_last_repay
    ["cur_doi_report"       , "day_itvl(PD01BR03, today)"       , None                              , "最近报告日期距今日"],# Mixed: mixed_doi_report
]
PBOC_ACC_INFO_MONTHLY = [
    # 月度表现
    ["monthly_acc_status"       , "map(PD01CD01, d1_acc_status)"    , "acc_cat == 1"                    , "月度状态"],          # Mixed: mixed_acc_status
    ["monthly_acc_status"       , "map(PD01CD01, r4_acc_status)"    , "acc_cat == 2"                    , "月度状态"],          # Mixed: mixed_acc_status
    ["monthly_acc_status"       , "map(PD01CD01, r1_acc_status)"    , "acc_cat == 3"                    , "月度状态"],          # Mixed: mixed_acc_status
    ["monthly_acc_status"       , "map(PD01CD01, r23_acc_status)"   , "(acc_cat >= 4) & (acc_cat <= 5)" , "月度状态"],          # Mixed: mixed_acc_status
    ["monthly_lvl5_status"      , "map(PD01CD02, lvl5_status)"      , None                              , "月度5级分类"],       # Mixed: mixed_lvl5_status
    ["monthly_doi_last_repay"   , "day_itvl(PD01CR03, today)"       , None                              , "月度还款距今日"],    # Mixed: mixed_doi_last_repay
    ["monthly_doi_report"       , "day_itvl(PD01CR01, today)"       , None                              , "月度报告日期距今日"],# Mixed: mixed_doi_report
    ["monthly_usd_ppor"         , "sdiv(PD01CJ02, acc_lmt)"         , None                              , "月度额度使用率"],
    ["last_6m_avg_usd"          , "cb_fst(PD01CJ12, PD01CJ13)"      , None                              , "R23账户最近6月平均透支额"],
    ["last_6m_max_usd"          , "cb_fst(PD01CJ14, PD01CJ15)"      , None                              , "R23账户最近6月最大透支额"],
]
PBOC_ACC_INFO_MIXED = [
    # 最新表现、月度表现根据说明文档混合
    ["mixed_acc_moi_folw"       , "mon_itvl(cb_fst(PD01BR01, PD01AR02), today)"         , None  , "账户关闭距今月"],
    ["mixed_acc_status"         , "cb_fst(cur_acc_status, monthly_acc_status)"          , None  , "账户状态"],
    ["mixed_lvl5_status"        , "cb_max(cur_lvl5_status, monthly_lvl5_status)"        , None  , "账户5级分类"],
    ["mixed_ots"                , "cb_min(PD01BJ01, PD01CJ01)"                          , None  , "账户余额"],
    ["mixed_last_repay_amt"     , "cb_fst(PD01BJ02, PD01CJ05)"                          , None  , "最近实还款"],
    ["mixed_doi_last_repay"     , "cb_min(cur_doi_last_repay, monthly_doi_last_repay)"  , None  , "最近还款距今日"],
    ["mixed_doi_report"         , "cb_min(cur_doi_report, monthly_doi_report)"          , None  , "报告时间"],
    # 按月应还 - 包含已结清
    ["alle_mon"                     , "PD01AS01"                                                    , "acc_repay_freq == 3" , "全部还款期数（月）"],
    ["alle_mon"                     , "cb_max(mon_itvl(cb_fst(PD01AR02, PD01BR01), PD01AR01), 1)"   , "acc_repay_freq != 3" , "全部还款期数（月）"],
    ["mixed_alle_monthly_repay"     , "sdiv(PD01AJ01, alle_mon)"                                    , "acc_cat <= 2"        , "D1R4全周期按月应还款"],
    # 按月应还
    ["folw_mon"                     , "PD01CS01"                                                    , "acc_repay_freq == 3" , "剩余还款期数（月）"],
    ["folw_mon"                     , "cb_max(mon_itvl(PD01AR02, cb_max(PD01CR01, PD01CR04)), 1)"   , "acc_repay_freq != 3" , "剩余还款期数（月）"],
    # D1R41 月负债：按月还款账户直接取 `PD01CJ04-本月应还款`，否则直接按月直接除
    ["mixed_folw_monthly_repay_"    , "cb_max(PD01CJ04, sdiv(PD01CJ01, folw_mon))"                  , "acc_cat <= 3"        , "D1R41按月应还款"],
    ["mixed_folw_monthly_repay"     , "cb_fst(mixed_folw_monthly_repay_, mixed_alle_monthly_repay)" , "acc_cat <= 3"        , "D1R41按月应还款"],
    ["mixed_folw_monthly_repay"     , "cb_max(PD01CJ04, smul(PD01CJ12, 0.1))"                       , "acc_cat == 4"        , "R2按月应还款"],
]
PBOC_ACC_INFO = {
    "part": "pboc_acc_info",
    "trans": (PBOC_ACC_INFO_BASIC
              + PBOC_ACC_INFO_LATEST
              + PBOC_ACC_INFO_MONTHLY
              + PBOC_ACC_INFO_MIXED)
}


# %%
CREDIT_LMT_CAT = {
    "10": (10       , "循环贷款额度"),
    "20": (20       , "非循环贷款额度"),
    "30": (30       , "信用卡共享额度"),
    "31": (31       , "信用卡独立额度"),
}
CREDIT_PROTOCAL_STATUS = {
    "1": (1         , "有效"),
    "2": (2         , "到期/失效"),
}

PBOC_CREDIT_INFO = {
    "part": "pboc_credit_info",
    "trans": [
        ["credit_org_cat", "map(PD02AD01, org_cat)", None, "授信账户管理机构类型"],
        ["credit_cat"   , "map(PD02AD02, credit_lmt_cat)", None, "授信额度类型"],
        ["credit_status", "map(PD02AD04, credit_protocal_status)", None, "授信协议状态"],
        ["credit_moi_start", "mon_itvl(PD02AR01, today)", None, "授信起始距今月"],
        ["credit_moi_end", "mon_itvl(PD02AR02, today)", None, "授信截至距今月"],
    ],
}


def credit_cat(field: str = "credit_cat"):
    reprs = [
        ("rev", f"{field} == 10", "循环贷授信额度"),
        ("norev", f"{field} == 20", "非循环贷授信额度"),
        ("card", f"{field} >= 30", "信用卡独立、共享授信额度"),
    ]
    return reprs


# %%
REL_BORROWER_TYPE = {
    "1": (1         , "自然人"),
    "2": (2         , "组织机构"),
}
REL_RESP_TYPE = {
    "1": (1         , "共同借款人"),
    "2": (2         , "保证人"),
    "3": (3         , "票据承兑人"),
    "4": (4         , "应收账款债务人"),
    "5": (5         , "供应链中核心企业"),
    "9": (9         , "其他"),
}
# 企业业务种类，仅相关还款责任涉及
COMP_BIZ_CAT = {
    "10": (10       , "企业债"),
    "11": (11       , "贷款"),
    "12": (12       , "贸易融资"),
    "13": (13       , "保理融资"),
    "14": (14       , "融资租赁"),
    "15": (15       , "证券类融资"),
    "16": (16       , "透支"),
    "21": (21       , "票据贴现"),
    "31": (31       , "黄金借贷"),
    "41": (41       , "垫款"),
    "51": (51       , "资产处置"),
}
PBOC_REL_INFO = {
    "part": "pboc_rel_info",
    "trans": [
        ["rel_org_cat", "map(PD03AD01, org_cat)", None, "相关还款责任管理机构类型"],
        ["rel_biz_cat", "map(PD03AD02, biz_cat)",
         'PD03AD08 == "1"', "相关还款责任业务类型"],
        ["rel_biz_cat", "map(PD03AD02, comp_biz_cat)" ,
         'PD03AD08 == "2"', "相关还款责任业务类型"],
        ["rel_lvl5_status", "map(PD03AD05, lvl5_status)", None, "相关还款责任5级分类"],
        ["rel_repay_status", "map(PD03AD07, repay_status)", None, "相关还款责任还款状态"],
        ["rel_moi_start", "mon_itvl(PD03AR01, today)", None, "相关还款责任起始距今"],
        ["rel_moi_end", "mon_itvl(PD03AR02, today)", None, "相关还款责任截至距今"],
    ],
}


# %%
INQ_REASON_CAT = {
    # 1x: 贷前、保前、授信首批
    # 2x: 贷后、保后管理
    # 3x: 关联业务审查
    # 4x: 司法调查
    "01": (21       , "贷后管理"),
    "02": (11       , "贷款审批"),
    "03": (12       , "信用卡审批"),
    "08": (13       , "担保资格审查"),
    "09": (41       , "司法调查"),
    "16": (14       , "公积金提取复核查询"),
    "18": (15       , "股指期货开户"),
    "19": (31       , "特约商户实名审查"),
    "20": (16       , "保前审查"),
    "21": (22       , "保后管理"),
    "22": (32       , "法人代表、负责人、高管等资信审查"),
    "23": (33       , "客户准入资格审查"),
    "24": (17       , "融资审批"),
    "25": (34       , "资信审查"),
    "26": (18       , "额度审批"),
}


PBOC_INQ_REC = {
    "part": "pboc_inq_rec",
    "trans": [
        ["inq_rec_org_cat", "map(PH010D01, org_cat)", None, "查询机构类型"],
        ["inq_rec_reason_cat", "map(PH010Q03, inq_reason_cat)", None, "查询原因类型"],
        ["inq_rec_moi", "mon_itvl(PH010R01, today)", None, "查询距今月"],
        ["inq_rec_doi", "day_itvl(PH010R01, today)", None, "查询距今月"],
    ],
}


def inq_rec_reason_cat(field: str = "inq_rec_reason_cat"):
    reprs = [
        ("for_pre", f"{field} < 20", "贷前审批"),
        ("for_after", f"({field} > 20) & ({field} < 30)", "贷后管理"),
        ("for_rel", f"({field} > 30) & ({field} < 40)", "关联审查"),
        ("for_others", f"{field} > 40", "其他原因审查")
    ]
    reprs += [
        ("for_loan", f"{field} == 11", "贷前审批_贷款"),
        ("for_card", f"{field} == 12", "贷前审批_信用卡"),
        ("for_guar", f"{field} == 13", "贷前审批_担保资格审查"),
        ("for_leasing", f"{field} == 17", "贷前审批_融资审批"),
        ("for_lmt", f"{field} == 18", "贷前审批_额度审批"),
        ("for_prev2", f"isin({field}, [11, 12, 16, 17])", "贷前审批"),
    ]
    reprs += [(None, None, None)]
    return reprs


# %%
POSTFEE_ACC_CAT_M = {
    "TE": (1        , "电信缴费账户"),
    "UE": (2        , "公用事业缴费账户"),
}
POSTFEE_ACC_STATUS_M = {
    "0": (0         , "正常"),
    "1": (1         , "欠费"),
}

PBOC_POSTFEE_INFO = {
    "part": "pboc_postfee_info",
    "trans": [
        ["postfee_acc_cat", "map(PE01AD01, postfee_acc_cat_m)", None, "后付费账户类型"],
        ["postfee_acc_status", "map(PE01AD03, postfee_acc_status_m)", None, "账户状态"],
    ],
}


def postfee_acc_cat(field: str = "postfee_acc_cat"):
    reprs = [
        ("tel", f"{field} == 1", "电信业务"),
        ("pub", f"{field} == 2", "水电费等公共事业"),
        (None, None, None),
    ]
    return reprs


def postfee_acc_status(field: str = "postfee_acc_status"):
    reprs = [
        ("nor", f"{field} == 0", "正常"),
        ("ovd", f"{field} == 1", "欠费"),
        (None, None, None),
    ]
    return reprs


# %%
# 人行征信不再更新公积金信息
HOUSING_FUND_STATUS = {
    "1": (1         , "缴交"),
    "2": (2         , "封存"),
    "3": (3         , "销户"),
}

PBOC_HOUSING_FUND = {
    "part": "pboc_housing_fund",
    "trans":[
        ["hf_status", "map(PF05AD01, housing_fund_status)", None, "缴交状态"],
    ],
}


def housing_fund_status(field: str = "hf_status"):
    reprs = [
        ("active", "hf_status == 1", "缴交"),
        ("frozen", "hf_status == 2", "封存"),
        ("closed", "hf_status == 3", "销户"),
        (None, None, None),
    ]
    return reprs


# %%
# 报送、查询码表不完全相同
EDU_RECORD = {
    "10": (10       , "研究生"),
    "20": (20       , "本科"),
    "30": (30       , "大专"),
    "40": (40       , "中专、职高、技校"),
    "60": (60       , "高中"),
    "70": (70       , "初中"),  # 仅采集码表
    "80": (80       , "小学"),  # 仅采集码表
    "90": (90       , "其他"),
    "91": (91       , "初中及以下"),
    "99": (99       , "未知"),
}
EDU_DEGREE = {
    "1": (1         , "名誉博士"),
    "2": (2         , "博士"),
    "3": (3         , "硕士"),
    "4": (4         , "学士"),
    "5": (5         , "无"),
    "9": (9         , "未知"),
    "0": (0         , "其他"),
}
EMP_STATUS = {
    "11": (11       , "国家公务员"),
    "13": (13       , "专业技术人员"),
    "17": (17       , "职员"),
    "21": (21       , "企业管理人员"),
    "24": (24       , "工人"),
    "27": (27       , "农民"),
    "31": (31       , "学生"),
    "37": (37       , "现役军人"),
    "51": (51       , "自有职业者"),
    "54": (54       , "个体经营者"),
    "70": (70       , "无业人员"),
    "80": (80       , "退（离）休人员"),
    "90": (90       , "其他"),
    "91": (91       , "在职"),
    "99": (99       , "未知"),
}
MARITAL_STATUS = {
    "10": (10       , "未婚"),
    "20": (20       , "已婚"),
    "30": (30       , "丧偶"),
    "40": (40       , "离婚"),
    "91": (91       , "单身"),
    "99": (99       , "未知"),
}


def marital_status(field: str = "marital_status"):
    reprs = [
        ("married", f"{field} == 20", "已婚"),
        ("uncped", f"{field} != 20", "非已婚"),
    ]
    return reprs


# %%
RES_STATUS = {
    "1": (11        , "自置"),
    "2": (12        , "按揭"),
    "3": (21        , "亲属楼宇"),
    "4": (22        , "集体宿舍"),
    "5": (31        , "租房"),
    "6": (14        , "共有住宅"),
    "7": (98        , "其他"),
    "11": (13       , "自有"),
    "12": (23       , "借住"),
    "9": (99        , "未知"),
}

PBOC_ADDRESS = {
    "part": "pboc_address",
    "trans": [
        ["pi_res_status", "map(PB030D01, res_status)", None, "居住状况"],
    ],
}


def res_status(field: str = "pi_res_status"):
    res_status = [
        ("owned", f"{field} < 20", "自有、按揭、共有"),
        ("owned_all", f"({field} > 10) & ({field} <= 13)", "自置、按揭"),
        ("rented", f"({field} > 20) & ({field} < 40)", "租住、借住"),
        ("other", f"{field} > 90", "其他"),
    ]
    return res_status


# %%
COMP_CHARACTER = {
    "10": (10       , "机关、事业单位"),
    "20": (20       , "国有企业"),
    "30": (30       , "外资企业"),
    "40": (40       , "个体、私营企业"),
    "50": (50       , "其他（三资企业、民营企业、民间团体）"),
    "99": (99       , "未知"),
}


def comp_char(field: str = "pi_comp_char"):
    comp_char = [
        ("char_gov", f"{field} == 10", "机关、事业单位"),
        ("char_gov_cap", f"{field} == 20", "国有企业"),
        ("char_fogn_cap", f"{field} == 30", "外资企业"),
        ("char_priv_cap", f"{field} == 40", "个体、私营企业"),
        ("char_other_cap", f"{field} == 50", "其他（三资、民营、团体）"),
        ("char_other", f"{field} == 99", "未知"),
    ]
    return comp_char


# %%
COMP_INDUSTRY = {
    "A": (11        , "农、林、牧、渔业"),
    "B": (21        , "采矿业"),
    "C": (22        , "制造业"),
    "D": (23        , "电力、热力、燃气及水生产和供应业"),
    "E": (24        , "建筑业"),
    "F": (31        , "批发和零售业"),
    "G": (32        , "交通运输、仓储和邮储业"),
    "H": (33        , "住宿和餐饮业"),
    "I": (34        , "信息传输、软件和信息技术服务业"),
    "J": (35        , "金融业"),
    "K": (36        , "房地产业"),
    "L": (37        , "租赁和商务服务业"),
    "M": (38        , "科学研究和技术服务业"),
    "N": (39        , "水利、环境和公共设施管理业"),
    "O": (40        , "居民服务、修理和其他服务业"),
    "P": (41        , "教育"),
    "Q": (42        , "卫生和社会工作"),
    "R": (43        , "文化、体育和娱乐业"),
    "S": (44        , "公共管理、社会保障和社会组织"),
    "T": (45        , "国际组织"),
    "9": (99        , "未知"),
}


def comp_indust(field: str = "pi_comp_indust"):
    comp_indust = [
        ("1indust", f"{field} < 20", "第一产业"),
        ("2indust", f"({field} > 20) & ({field} < 30)", "第二产业"),
        ("3indust", f"({field} > 30) & ({field} < 90)", "第三产业"),
        ("other_ind", f"{field} == 99", "其他"),
    ]
    return comp_indust


# %%
# 职业
COMP_PROFESSION = {
    "0": (11        , "国家机关、党群组织、企业、事业单位负责人"),
    "1": (31        , "专业技术人员"),
    "3": (41        , "办事人员和有关人员"),
    "4": (42        , "商业、服务业人员"),
    "5": (21        , "农、林、牧、渔、水利业生产人员"),
    "6": (32        , "生产、运输设备操作人员及有关人员"),
    "X": (12        , "军人"),
    "Y": (98        , "不便分类的其他从业人员"),
    "Z": (99        , "未知"),
}


def comp_prof(field: str = "pi_comp_prof"):
    comp_prof = [
        ("prof_head", f"{field} == 11", "国家机关、党群组织、企业、事业单位负责人"),
        ("prof_soldier", f"{field} == 12", "军人"),
        ("prof_prod", f"{field} == 21", "生产人员"),
        ("prof_prof", f"({field} > 30) & ({field} < 40)", "技术人员"),
        ("prof_serv", f"({field} > 40) & ({field} < 50)", "服务人员"),
        ("prof_other", f"{field} > 90", "其他职业"),
    ]
    return comp_prof


# %%
# 职务
COMP_POSITION = {
    "1": (1         , "高级领导"),
    "2": (2         , "中级领导"),
    "3": (3         , "一般员工"),
    "4": (98        , "其他"),
    "9": (99        , "未知"),
}


def comp_pos(field: str = "pi_comp_pos"):
    comp_pos = [
        ("pos_sup", f"{field} == 1", "高级领导"),
        ("pos_mid", f"{field} == 2", "中级领导"),
        ("pos_inf", f"{field} == 3", "一般员工"),
        ("pos_other", f"{field} > 90", "其他职务"),
    ]
    return comp_pos


# 职称
COMP_PROF_TITLE = {
    "0": (98        , "无"),
    "1": (1         , "高级"),
    "2": (2         , "中级"),
    "3": (3         , "初级"),
    "9": (99        , "未知"),
}


def comp_prof_title(field: str = "pi_comp_prof_title"):
    comp_prof_title = [
        ("pos_title_sup", f"{field} == 1", "高级职称"),
        ("pos_title_mid", f"{field} == 2", "中级职称"),
        ("pos_title_low", f"{field} == 3", "初级职称"),
        ("pos_title_none", f"{field} > 90", "无职称"),
    ]
    return comp_prof_title


# %%
PBOC_COMPANY = {
    "part": "pboc_company",
    "trans": [
        ["pi_comp_job", "map(PB040D01, emp_status)", None, "就业状况"],
        ["pi_comp_char", "map(PB040D02, comp_character)", None, "单位性质"],
        ["pi_comp_indust", "map(PB040D03, comp_industry)", None, "行业"],
        ["pi_comp_prof", "map(PB040D04, comp_profession)", None, "职业"],
        ["pi_comp_pos", "map(PB040D05, comp_position)", None, "职务"],
        ["pi_comp_prof_title", "map(PB040D06, comp_prof_title)", None, "职称"],
    ],
}


# %%
MAPPERS = {
    "repay_status"                      : REPAY_STATUS,
    "repay_status_spec"                 : REPAY_STATUS_SPEC,
    "special_trans_type"                : SPECIAL_TRANS_TYPE,
    "special_accd_type"                 : SPECIAL_ACCD_TYPE,
    "biz_cat"                           : BIZ_CAT,
    "comp_biz_cat"                      : COMP_BIZ_CAT,
    "cdr_cat"                           : CDR_CAT,
    "org_cat"                           : ORG_CAT,
    "trans_status"                      : TRANS_STATUS,
    "c1_acc_status"                     : C1_ACC_STATUS,
    "d1_acc_status"                     : D1_ACC_STATUS,
    "r4_acc_status"                     : R4_ACC_STATUS,
    "r1_acc_status"                     : R1_ACC_STATUS,
    "r23_acc_status"                    : R23_ACC_STATUS,
    "lvl5_status"                       : LVL5_STATUS,
    "credit_lmt_cat"                    : CREDIT_LMT_CAT,
    "credit_protocal_status"            : CREDIT_PROTOCAL_STATUS,
    "postfee_acc_cat_m"                 : POSTFEE_ACC_CAT_M,
    "postfee_acc_status_m"              : POSTFEE_ACC_STATUS_M,
    "inq_reason_cat"                    : INQ_REASON_CAT,
    "repay_type"                        : REPAY_TYPE,
    "repay_freq"                        : REPAY_FREQ,
    "guar_type"                         : GUAR_TYPE,
    "rel_borrower_type"                 : REL_BORROWER_TYPE,
    "rel_resp_type"                     : REL_RESP_TYPE,
    "edu_record"                        : EDU_RECORD,
    "edu_degree"                        : EDU_DEGREE,
    "emp_status"                        : EMP_STATUS,
    "marital_status"                    : MARITAL_STATUS,
    "res_status"                        : RES_STATUS,
    "comp_character"                    : COMP_CHARACTER,
    "comp_industry"                     : COMP_INDUSTRY,
    "comp_profession"                   : COMP_PROFESSION,
    "comp_position"                     : COMP_POSITION,
    "comp_prof_title"                   : COMP_PROF_TITLE,
    "housing_fund_status"               : HOUSING_FUND_STATUS,
}

MAPPERS_CODE = {k: {kk: vv[0] for kk, vv in v.items()}
                for k, v in MAPPERS.items()}
MAPPERS_CHN = {k: {kk: vv[1] for kk, vv in v.items()}
               for k, v in MAPPERS.items()}


# %%
TRANS_CONF = {
    # LV2
    PBOC_ACC_REPAY_60_MONTHLY["part"]   : PBOC_ACC_REPAY_60_MONTHLY,
    PBOC_ACC_SPECIAL_ACCD["part"]       : PBOC_ACC_SPECIAL_ACCD,
    PBOC_ACC_SPECIAL_INSTS["part"]      : PBOC_ACC_SPECIAL_INSTS,
    PBOC_ACC_SPECIAL_TRANS["part"]      : PBOC_ACC_SPECIAL_TRANS,
    # LV1
    PBOC_ACC_INFO["part"]               : PBOC_ACC_INFO,
    PBOC_CREDIT_INFO["part"]            : PBOC_CREDIT_INFO,
    PBOC_REL_INFO["part"]               : PBOC_REL_INFO,
    PBOC_INQ_REC["part"]                : PBOC_INQ_REC,
    PBOC_HOUSING_FUND["part"]           : PBOC_HOUSING_FUND,
    PBOC_POSTFEE_INFO["part"]           : PBOC_POSTFEE_INFO,
    PBOC_ADDRESS["part"]                : PBOC_ADDRESS,
    PBOC_COMPANY["part"]                : PBOC_COMPANY,
}


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


# %%
#TODO
TRANS_EXTRACT_CONF = {
    "pboc_basic_info": [
        ["pinfo_gender"                         , "PB01AD01"                        , None  , "性别"],
        ["pinfo_edu"                            , "PB01AD02"                        , None  , "学历"],
        ["pinfo_edu_lvl"                        , "PB01AD03"                        , None  , "学位"],
        ["pinfo_job_status"                     , "PB01AD04"                        , None  , "就业状况"],
        ["pinfo_citizenship"                    , "PB01AD05"                        , None  , "国籍"],
        ["pinfo_email"                          , "PB01AQ01"                        , None  , "电子邮箱"],
        ["pinfo_com_addr"                       , "PB01AQ02"                        , None  , "通讯地址"],
        ["pinfo_household_addr"                 , "PB01AQ03"                        , None  , "户籍地址"],
        ["pinfo_marriage_status"                , "PB020D01"                        , None  , "婚姻状态"],
        ["pinfo_spouse_certno"                  , "PB020I01"                        , None  , "配偶证件号"],
        ["pinfo_spouse_name"                    , "PB020Q01"                        , None  , "配偶姓名"],
        ["pinfo_spouse_comp"                    , "PB020Q02"                        , None  , "配偶工作单位"],
        ["pinfo_mobile"                         , "PB020Q03"                        , None  , "配偶联系电话"],
    ],
    "pboc_biz_abst": [
        ["abst_loan_housing_prdtn_max"          , "month(PC02AR02_11 - today)"      , None  , "首笔住房贷款发放月份数"],
        ["abst_loan_housing_cnt"                , "PC02AS03_11"                     , None  , "住房贷款账户数"],
        ["abst_loan_chousing_prdtn_max"         , "month(PC02AR02_12 - today)"      , None  , "首笔商用房贷款发放月份数"],
        ["abst_loan_chousing_cnt"               , "PC02AS03_12"                     , None  , "商用房贷款账户数"],
        ["abst_loan_nonhousing_prdtn_max"       , "month(PC02AR02_19 - today)"      , None  , "首笔其他类贷款发放月份数"],
        ["abst_loan_nonhousing_cnt"             , "PC02AS03_19"                     , None  , "其他类贷款账户数"],
        ["abst_card_prdtn_max"                  , "month(PC02AR02_21 - today)"      , None  , "贷记卡首笔业务发放月份数"],
        ["abst_card_cnt"                        , "PC02AS03_21"                     , None  , "贷记卡账户数"],
        ["abst_qcard_prdtn_max"                 , "month(PC02AR02_22 - today)"      , None  , "准贷记卡首笔业务发放月份数"],
        ["abst_qcard_cnt"                       , "PC02AS03_22"                     , None  , "准贷记卡账户数"],
        ["abst_nonlc_prdtn_max"                 , "month(PC02AR02_99 - today)"      , None  , "首笔其他信贷账户业务发放月份数"],
        ["abst_nonlc_cnt"                       , "PC02AS03_99"                     , None  , "其他信贷账户业务账户数"],
        ["abst_acc_cnt"                         , "PC02AS01"                        , None  , "信贷账户数合计"],
        ["abst_acc_type_cnt"                    , "PC02AS02"                        , None  , "信贷业务类型数合计"],
        ["abst_assets_disp_cnt"                 , "PC02BS03_1"                      , None  , "资产处置业务账户数"],
        ["abst_assets_disp_ots_sum"             , "PC02BJ02_1"                      , None  , "资产处置业务账户余额"],
        ["abst_adv_cnt"                         , "PC02BS03_2"                      , None  , "垫款业务账户数"],
        ["abst_adv_ots_sum"                     , "PC02BJ02_2"                      , None  , "垫款业务账户余额"],
    ],
    "pboc_cacc_abst": [
        ["abst_c1acc_ots_sum"                   , "PC02BJ01"                        , None  , "C1类账户余额合计"],
        ["abst_c1acc_cnt"                       , "PC02BS01"                        , None  , "C1类账户数合计"],
        ["abst_c1acc_rec_cnt"                   , "PC02BS02"                        , None  , "C1类账户记录数"],
    ],
    "pboc_dum_abst": [
        ["abst_dum_ots_sum"                     , "PC02CJ01"                        , None  , "呆账账户余额"],
        ["abst_dum_cnt"                         , "PC02CS01"                        , None  , "呆账账户数"],
    ],
    "pboc_dracc_abst": [
        ["abst_d1acc_ovd_cnt"                   , "PC02DS02_1"                      , None  , "逾期D1账户数"],
        ["abst_d1acc_ovd_mcnt"                  , "PC02DS03_1"                      , None  , "逾期D1月份数"],
        ["abst_d1acc_ovd_max"                   , "PC02DJ01_1"                      , None  , "逾期D1单月最高逾期（透支）总额"],
        ["abst_d1acc_ovd_mcnt_max"              , "PC02DS04_1"                      , None  , "逾期D1最长逾期（透支）月数"],
        ["abst_d1acc_orgs"                      , "PC02ES01"                        , None  , "D1管理机构数"],
        ["abst_d1acc_cnt"                       , "PC02ES02"                        , None  , "D1账户数"],
        ["abst_d1acc_lmt_sum"                   , "PC02EJ01"                        , None  , "D1授信总额"],
        ["abst_d1acc_ots_sum"                   , "PC02EJ02"                        , None  , "D1余额"],
        ["abst_d1acc_duepay_last_6m"            , "PC02EJ03"                        , None  , "D1最近6个月平均应还款"],
        ["abst_r4acc_ovd_cnt"                   , "PC02DS02_2"                      , None  , "逾期R4账户数"],
        ["abst_r4acc_ovd_mcnt"                  , "PC02DS03_2"                      , None  , "逾期R4月份数"],
        ["abst_r4acc_ovd_max"                   , "PC02DJ01_2"                      , None  , "逾期R4单月最高逾期（透支）总额"],
        ["abst_r4acc_ovd_mcnt_max"              , "PC02DS04_2"                      , None  , "逾期R4最长逾期（透支）月数"],
        ["abst_r4acc_orgs"                      , "PC02FS01"                        , None  , "R4管理机构数"],
        ["abst_r4acc_cnt"                       , "PC02FS02"                        , None  , "R4账户数"],
        ["abst_r4acc_lmt_sum"                   , "PC02FJ01"                        , None  , "R4授信总额"],
        ["abst_r4acc_ots_sum"                   , "PC02FJ02"                        , None  , "R4余额"],
        ["abst_r4acc_duepay_last_6m"            , "PC02FJ03"                        , None  , "R4最近6个月平均应还款"],
        ["abst_r1acc_ovd_cnt"                   , "PC02DS02_3"                      , None  , "逾期R1账户数"],
        ["abst_r1acc_ovd_mcnt"                  , "PC02DS03_3"                      , None  , "逾期R1月份数"],
        ["abst_r1acc_ovd_max"                   , "PC02DJ01_3"                      , None  , "逾期R1单月最高逾期（透支）总额"],
        ["abst_r1acc_ovd_mcnt_max"              , "PC02DS04_3"                      , None  , "逾期R1最长逾期（透支）月数"],
        ["abst_r1acc_orgs"                      , "PC02GS01"                        , None  , "R1管理机构数"],
        ["abst_r1acc_cnt"                       , "PC02GS02"                        , None  , "R1账户数"],
        ["abst_r1acc_lmt_sum"                   , "PC02GJ01"                        , None  , "R1授信总额"],
        ["abst_r1acc_ots_sum"                   , "PC02GJ02"                        , None  , "R1余额"],
        ["abst_r1acc_duepay_last_6m"            , "PC02GJ03"                        , None  , "R1最近6个月平均应还款"],
        ["abst_r2acc_ovd_cnt"                   , "PC02DS02_4"                      , None  , "逾期R2账户数"],
        ["abst_r2acc_ovd_mcnt"                  , "PC02DS03_4"                      , None  , "逾期R2月份数"],
        ["abst_r2acc_ovd_max"                   , "PC02DJ01_4"                      , None  , "逾期R2单月最高逾期（透支）总额"],
        ["abst_r2acc_ovd_mcnt_max"              , "PC02DS04_4"                      , None  , "逾期R2最长逾期（透支）月数"],
        ["abst_r2acc_orgs"                      , "PC02HS01"                        , None  , "R2管理机构数"],
        ["abst_r2acc_cnt"                       , "PC02HS02"                        , None  , "R2账户数"],
        ["abst_r2acc_lmt_sum"                   , "PC02HJ01"                        , None  , "R2授信总额"],
        ["abst_r2acc_lmt_max_per_org"           , "PC02HJ02"                        , None  , "R2单家行最高授信额"],
        ["abst_r2acc_lmt_min_per_org"           , "PC02HJ03"                        , None  , "R2单家行最低授信额"],
        ["abst_r2acc_usd_sum"                   , "PC02HJ04"                        , None  , "R2已用额度"],
        ["abst_r2acc_usd_sum_avg_last_6m"       , "PC02HJ05"                        , None  , "R2最近6个月平均使用额度"],
        ["abst_r3acc_ovd_cnt"                   , "PC02DS02_5"                      , None  , "逾期R3账户数"],
        ["abst_r3acc_ovd_mcnt"                  , "PC02DS03_5"                      , None  , "逾期R3月份数"],
        ["abst_r3acc_ovd_max"                   , "PC02DJ01_5"                      , None  , "逾期R3单月最高逾期（透支）总额"],
        ["abst_r3acc_ovd_mcnt_max"              , "PC02DS04_5"                      , None  , "逾期R3最长逾期（透支）月数"],
        ["abst_r3acc_orgs"                      , "PC02IS01"                        , None  , "R3管理机构数"],
        ["abst_r3acc_cnt"                       , "PC02IS02"                        , None  , "R3账户数"],
        ["abst_r3acc_lmt_sum"                   , "PC02IJ01"                        , None  , "R3授信总额"],
        ["abst_r3acc_lmt_max_per_org"           , "PC02IJ02"                        , None  , "R3单家行最高授信额"],
        ["abst_r3acc_lmt_min_per_org"           , "PC02IJ03"                        , None  , "R3单家行最低授信额"],
        ["abst_r3acc_usd_sum"                   , "PC02IJ04"                        , None  , "R3已用额度"],
        ["abst_r3acc_usd_sum_avg_last_6m"       , "PC02IJ05"                        , None  , "R3最近6个月平均使用额度"],
    ],
    "pboc_postfee_abst": [
        ["abst_telacc_ovd_cnt"                  , "PC030S02_1"                      , None  , "电信业务欠费账户数"],
        ["abst_telacc_ovd_sum"                  , "PC030J01_1"                      , None  , "电信业务欠费金额"],
        ["abst_pubfair_ovd_cnt"                 , "PC030S02_2"                      , None  , "水电费等公共事业欠费账户数"],
        ["abst_pubfair_ovd_sum"                 , "PC030J01_2"                      , None  , "水电费等公共事业欠费金额"],
    ],
    "pboc_public_abst": [
        ["abst_taxs_ovd_cnt"                    , "PC040S02_1"                      , None  , "欠税信息记录数"],
        ["abst_taxs_sum_sum"                    , "PC040J01_1"                      , None  , "欠税信息涉及金额"],
        ["abst_lawsuit_cnt"                     , "PC040S02_2"                      , None  , "民事判决信息记录数"],
        ["abst_lawsuit_sum"                     , "PC040J01_2"                      , None  , "民事判决信息涉及金额"],
        ["abst_enforce_cnt"                     , "PC040S02_3"                      , None  , "强制执行信息记录数"],
        ["abst_enforce_sum"                     , "PC040J01_3"                      , None  , "强制执行信息涉及金额"],
        ["abst_gov_punishment_cnt"              , "PC040S02_4"                      , None  , "行政处罚信息记录数"],
        ["abst_gov_punishment_sum"              , "PC040J01_4"                      , None  , "行政处罚信息涉及金额"],
    ],
    "pboc_inq_abst": [
        ["abst_loan_auth_inq_org_cnt_last_1m"   , "PC05BS01"                        , None  , "最近1个月内的查询机构数(贷款审批)"],
        ["abst_card_auth_inq_org_cnt_last_1m"   , "PC05BS02"                        , None  , "最近1个月内的查询机构数(信用卡审批)"],
        ["abst_loan_auth_inq_cnt_last_1m"       , "PC05BS03"                        , None  , "最近1个月内的查询次数(贷款审批)"],
        ["abst_card_auth_inq_cnt_last_1m"       , "PC05BS04"                        , None  , "最近1个月内的查询次数(信用卡审批)"],
        ["abst_self_inq_cnt_last_1m"            , "PC05BS05"                        , None  , "最近1个月内的查询次数(本人查询)"],
        ["abst_after_loan_inq_cnt_last_24m"     , "PC05BS06"                        , None  , "最近2年内的查询次数(贷后管理)"],
        ["abst_guar_inq_cnt_last_24m"           , "PC05BS07"                        , None  , "最近2年内的查询次数(担保资格审查)"],
        ["abst_biz_auth_inq_cnt_last_24m"       , "PC05BS08"                        , None  , "最近2年内的查询次数(特约商户实名审查)"],
        ["abst_inq_prd_min"                     , "month(today - PC05AR01)"         , None  , "上次查询月份数"],
    ],
}
