#!/usr/bin/env python3
# ----------------------------------------------------------
#   Name: ka.py
#   Author: xyy15926
#   Created: 2022-11-10 21:44:59
#   Updated: 2024-03-03 19:34:48
#   Description:
# ----------------------------------------------------------

# %%
from __future__ import annotations

import pandas as pd
from itertools import product


# %%
def prod_agg_conds(
    aggs: list,
    cond_prods: list,
    varn_fmt: str = "{}_{}",
) -> pd.DataFrame:
    conds = []
    for agg_varn, agg_fn, agg_cmt in aggs:
        for cond_grp in product(*cond_prods):
            cond_varn, cond_cond, cond_cmt = [
                list(filter(lambda x: x is not None, ele))
                for ele in zip(*cond_grp)
            ]
            cond_varn_str = "_".join(cond_varn)
            cond_cond_str = " & ".join([f"({ele})" for ele in cond_cond])
            cond_cmt_str = "".join(cond_cmt)

            conds.append((varn_fmt.format(cond_varn_str, agg_varn),
                          cond_cond_str, agg_fn,
                          cond_cmt_str + agg_cmt))

    return pd.DataFrame.from_records(conds, columns=["key", "conds", "aggs", "cmt"])


# %%
def gen_agg_confs(confs):
    pconfs = []
    aconfs = {}
    for pname, pconf in confs.items():
        pname = pconf["part"]
        pconfs.append((pconf["part"], pconf["level"], pconf["from_"],
                       pconf["group_key"], pconf["join_key"]))
        conf_dfs = []
        for aggs, conds in pconf["cros"]:
            conds_ = []
            for cn in conds:
                # For condition indicator with format: (key, sub_key, sub_key,...)
                if isinstance(cn, (tuple, list)):
                    cn, *cnk = cn
                # For condition indicator with format: "key.[sub_key,sub_key]"
                else:
                    cn, *cnk = cn.split(".")
                    if cnk:
                        cnk = cnk[0][1: -1].split(",")

                acond = pconf["conds"][cn]
                if cnk:
                    scond = [acond[cnko] for cnko in cnk]
                    conds_.append(scond)
                elif isinstance(acond, dict):
                    conds_.append(list(acond.values()))
                else:
                    conds_.append(acond)

            aggs_ = [pconf["aggs"][an] for an in aggs]
            df = prod_agg_conds(aggs_, conds_, pconf["key_fmt"])
            conf_dfs.append(df)
        cdf = pd.concat(conf_dfs, axis=0)
        aconfs[pname] = cdf

    pconfs = pd.DataFrame.from_records(pconfs, columns=["part", "level", "from_",
                                                        "group_key", "join_key"])
    aconfs = pd.concat(aconfs.values(), keys=aconfs.keys()).droplevel(level=1)
    aconfs.index.name = "part"

    return pconfs, aconfs.reset_index()


# %%
def gen_confs():
    lv2_pconfs, lv2_aconfs = gen_agg_confs(LV2_AGG_CONF)
    lv1_pconfs, lv1_aconfs = gen_agg_confs(LV1_AGG_CONF)

    def from_lower_agg(conf_df, agg_repr:str):
        agg_D = {
            "max": ("max", "max({})", "最大值"),
            "sum": ("sum", "sum({})", "之和"),
        }
        varn_str, agg_fmt, cmt_str = agg_D[agg_repr]
        new_df = pd.DataFrame()
        new_df["key"] = conf_df["key"] + "_" + varn_str
        new_df["conds"] = None
        new_df["aggs"] = conf_df["key"].apply(lambda x: agg_fmt.format(x))
        new_df["cmt"] = conf_df["cmt"] + cmt_str

        return new_df

    acc_cat_cond_D = {key: val[1] for key, val in
                      LV1_AGG_CONF["acc_cat_info"]["conds"]["acc_cat"].items()}
    aggfn_nullp_agg = [
        ("acc_repay_60m", ("d1r41", "r23")),
        ("acc_special_trans", ("c1", "d1r41", "r23")),
        ("acc_special_accd", ("r2", )),
        ("acc_special_insts", ("r2", )),
    ]

    # 月度还款、特殊交易、特殊事件、大额分期
    conf_dfs = []
    lv20_pconfs = []
    for lpart, acc_cats in aggfn_nullp_agg:
        lower_confs = lv2_aconfs[lv2_aconfs["part"] == lpart]
        part = f"{lpart}_agg"
        lv20_pconfs.append((part, 0, f"{lpart},pboc_acc_info", None, None))
        for acc_cat in acc_cats:
            for aggstr, aaggs in [("cnt", ("sum", "max")),
                                  ("sum", ("sum", "max")),
                                  ("max", ("max",))]:
                part_conf = lower_confs[lower_confs["key"].str.endswith(aggstr)]
                for agg_repr in aaggs:
                    part_agg_df = from_lower_agg(part_conf, agg_repr)
                    part_agg_df["part"] = part
                    part_agg_df["key"] = acc_cat + part_agg_df["key"]
                    part_agg_df["conds"] = acc_cat_cond_D[acc_cat]
                    part_agg_df["cmt"] = acc_cat + part_agg_df["cmt"]
                    conf_dfs.append(part_agg_df)

    lower_df = pd.concat(conf_dfs, axis=0)
    lv20_pconfs = pd.DataFrame.from_records(lv20_pconfs,
                                            columns=["part", "level", "from_",
                                                     "join_key", "group_key"])

    pconfs = pd.concat([lv2_pconfs, lv1_pconfs, lv20_pconfs], axis=0)
    aconfs = pd.concat([lv2_aconfs, lv1_aconfs, lower_df], axis=0)

    return pconfs, aconfs


# %%
MAPPERS = {
    "repay_status": {
        "1": 1,         # 逾期1-30天
        "2": 2,         # 逾期31-60天
        "3": 3,         # 逾期61-90天
        "4": 4,         # 逾期91-120天
        "5": 5,         # 逾期121-150天
        "6": 6,         # 逾期151-180天
        "7": 7,         # 逾期180天以上
        "*": 0,         # 当月不需要还款且之前没有拖欠
        "#": 0,         # 未知
        "/": 0,         # 跳过
        "A": 0,         # 账单日调整,当月不出单
        "B": 7,         # 呆账
        "C": 0,         # 结清、正常销户
        "D": 1,         # 担保人代还
        "G": 7,         # （非正常销户）结束
        "M": 0,         # 约定还款日后月底前还款
        "N": 0,         # 正常还款
        "Z": 1,         # 以资抵债
    },
    "exchange_rate": {
        "USD": 7,
        "EUR": 7.7,
        "JPY": 0.05,
        "CNY": 1,
        "AUD": 4.7,
        "RUB": 0.07,
        "CAD": 5.3,
    },
    "special_trans_type": {
        # 性质逐渐恶劣
        "1": 1,         # 展期
        "2": 4,         # 担保人(第三方)代偿
        "3": 4,         # 以资抵债
        "4": 0,         # 提前还款
        "5": 0,         # 提前结清
        "6": 4,         # 强制平仓,未结清
        "7": 3,         # 强制平仓,已结清
        "8": 4,         # 司法追偿
        "9": 1,         # 其他
        "11": 2,        # 债务减免
        "12": 2,        # 资产剥离
        "13": 2,        # 资产转让
        "14": 1,        # 信用卡个性化分期
        "16": 1,        # 落实金融纾困等政策银行主动延期
        "17": 3,        # 强制平仓
    },
    "biz_cat": {
        # 贷款类业务
        "11": 111,      # 个人住房商业贷款
        "12": 112,      # 个人商用房(含商住两用)贷款
        "13": 113,      # 个人住房公积金贷款
        "21": 121,      # 个人汽车消费贷款
        "31": 131,      # 个人助学贷款
        "32": 132,      # 国家助学贷款
        "33": 133,      # 商业助学贷款
        "41": 141,      # 个人经营性贷款
        "42": 142,      # 个人创业担保贷款
        "51": 143,      # 农户贷款
        "52": 144,      # 经营性农户贷款
        "53": 151,      # 消费性农户贷款
        "91": 152,      # 其他个人消费贷款
        "99": 161,      # 其他贷款
        # 信用卡类业务
        "71": 211,      # 准贷记卡
        "81": 221,      # 贷记卡
        "82": 231,      # 大额专项分期卡
        # 证券融资类业务
        "61": 311,      # 约定购回式证券交易
        "62": 321,      # 股票质押式回购交易
        "63": 331,      # 融资融券业务
        "64": 341,      # 其他证券类融资
        # 融资租赁业务
        "92": 411,      # 融资租赁业务
        # 资产处置
        "A1": 511,      # 资产处置
        # 代偿
        "B1": 611,      # 代偿债务
    },
    "cdr_cat": {
        "D1": 1,        # 非循环贷账户
        "R1": 3,        # 循环贷账户
        "R2": 4,        # 贷记卡账户
        "R3": 5,        # 准贷记卡账户
        "R4": 2,        # 循环额度下分账户
        "C1": 99,       # 催收账户
    },
    "org_cat": {
        # 银行
        "11": 11,       # 商业银行
        "12": 12,       # 村镇银行
        "14": 14,       # 住房储蓄银行
        "15": 15,       # 外资银行
        # 非银金融机构
        "16": 27,       # 财务公司
        "21": 21,       # 信托公司
        "22": 22,       # 融资租赁公司
        "23": 23,       # 汽车金融公司
        "24": 24,       # 消费金融公司
        "25": 25,       # 贷款公司
        "26": 26,       # 金融资产管理公司
        "31": 31,       # 证券公司
        "41": 41,       # 保险公司
        "51": 51,       # 小额贷款公司
        "53": 53,       # 融资担保公司
        "54": 54,       # 保理公司
        # 其他机构
        "52": 98,       # 公积金管理中心
        "99": 99,       # 其他机构

    },
    "trans_status": {
        "0": 0,         # 债务人即将违约时自动垫款
        "1": 1,         # 逾期1-30天
        "2": 2,         # 逾期31-60天
        "3": 3,         # 逾期61-90天
        "4": 4,         # 逾期91-120天
        "5": 5,         # 逾期121-150天
        "6": 6,         # 逾期151-180天
        "7": 7,         # 逾期180天以上
        "9": 0,         # 未知
    },
    # 0: 结清、关闭、未激活
    # 1x: 正常
    # 2x: 逾期、催收、追偿
    # 3x: 平仓、止付、冻结、转出
    # 99: 呆账
    "c1_acc_status": {
        "1": 22,    # 催收
        "2": 0,     # 结束
    },
    "d1_acc_status": {
        "1": 11,    # 正常
        "2": 21,    # 逾期
        "3": 0,     # 结清
        "4": 99,    # 呆账
        "5": 36,    # 转出
        "6": 31,    # 担保物不足
        "7": 32,    # 强制平仓
        "8": 23,    # 司法追偿
    },
    "r4_acc_status": {
        "1": 11,    # 正常
        "2": 21,    # 逾期
        "3": 0,     # 结清
        "4": 99,    # 呆账
        "6": 31,    # 担保物不足
        "8": 23,    # 司法追偿
    },
    "r1_acc_status": {
        "1": 11,    # 正常
        "2": 21,    # 逾期
        "3": 0,     # 结清
        "4": 99,    # 呆账
        "5": 33,    # 银行止付
        "6": 31,    # 担保物不足
        "8": 23,    # 司法追偿
    },
    "r23_acc_status": {
        "1": 11,    # 正常
        "2": 35,    # 冻结
        "3": 34,    # 止付
        "31": 33,   # 银行止付
        "4": 0,     # 销户
        "5": 99,    # 呆账
        "6": 0,     # 未激活
        "8": 23,    # 司法追偿
    },
    "lvl5_status": {
        "1": 1,         # 正常
        "2": 2,         # 关注
        "3": 3,         # 次级
        "4": 4,         # 可疑
        "5": 5,         # 损失
        "6": 6,         # 违约
        "9": 1,         # 未分类
    },
    "postfee_cat": {
        "TE": 1,        # 电信缴费账户
        "UE": 2,        # 公用事业缴费账户
    },
    "postfee_status": {
        "0": 0,         # 正常
        "1": 1,         # 欠费
    },
    "inq_reason_cat": {
        # 1x: 贷前、保前、授信首批
        # 2x: 贷后、保后管理
        # 3x: 关联业务审查
        # 4x: 司法调查
        "1": 21,        # 贷后管理
        "2": 11,        # 贷款审批
        "3": 12,        # 信用卡审批
        "8": 13,        # 担保资格审查
        "9": 41,        # 司法调查
        "16": 14,       # 公积金提取复核查询
        "18": 15,       # 股指期货开户
        "19": 31,       # 特约商户实名审查
        "20": 16,       # 保前审查
        "21": 22,       # 保后管理
        "22": 32,       # 法人代表、负责人、高管等资信审查
        "23": 33,       # 客户准入资格审查
        "24": 17,       # 融资审批
        "25": 34,       # 资信审查
        "26": 18,       # 额度审批
    },
    "postfee_acc_cat_M": {
        "TE": 1,        # 电信缴费账户
        "UE": 2,        # 公共事业缴费账户
    },
    "postfee_acc_status_M": {
        "1": 1,         # 正常
        "0": 0,         # 欠费
    },
}


# %%
TRANS_CONF = {
    "pboc_acc_repay_60_monthly": [
        ["acc_repay_status"             , "map(PD01ED01, repay_status)"             , None],
        ["acc_repay_moi"                , "mon_itvl(PD01ER03, today)"               , None]
    ],
    "pboc_acc_special_trans": [
        ["acc_special_trans_type"       , "map(PD01FD01, special_trans_type)"       , None],
        ["acc_special_trans_moi"        , "mon_itvl(PD01FR01, today)"               , None],
    ],
    "pboc_acc_special_accd": [
        ["acc_special_accd_moi"         , "mon_itvl(PD01GR01, today)"               , None],
    ],
    "pboc_acc_special_insts": [
        ["acc_special_insts_moi_start"  , "mon_itvl(PD01HR01, today)"               , None],
        ["acc_special_insts_moi_end"    , "mon_itvl(PD01HR02, today)"               , None],
    ],
    "pboc_acc_info": [
        # 账户信息
        ["acc_cat"                      , "map(PD01AD01, cdr_cat)"                  , None],
        ["acc_org_cat"                  , "map(PD01AD02, org_cat)"                  , None],
        ["acc_biz_cat"                  , "map(PD01AD03, biz_cat)"                  , None],
        ["acc_trans_status"             , "map(PD01AD10, trans_status)"             , None],
        ["acc_lmt"                      , "cb_fst(PD01AJ01, PD01AJ02)"              , None],
        ["acc_moi_start"                , "mon_itvl(PD01AR01, today)"               , None],
        ["acc_moi_end"                  , "mon_itvl(PD01AR02, today)"               , None],
        ["last_6m_avg_usd"              , "cb_fst(PD01CJ12, PD01CJ13)"              , None],
        ["last_6m_max_usd"              , "cb_fst(PD01CJ14, PD01CJ15)"              , None],
        # 最新表现
        ["cur_acc_status"               , "map(PD01BD01, c1_acc_status)"            , "acc_cat == 99"],
        ["cur_acc_status"               , "map(PD01BD01, d1_acc_status)"            , "acc_cat == 1"],
        ["cur_acc_status"               , "map(PD01BD01, r4_acc_status)"            , "acc_cat == 2"],
        ["cur_acc_status"               , "map(PD01BD01, r1_acc_status)"            , "acc_cat == 3"],
        ["cur_acc_status"               , "map(PD01BD01, r23_acc_status)"           , "(acc_cat >= 4) & (acc_cat <= 5)"],
        ["cur_lvl5_status"              , "map(PD01BD03, lvl5_status)"              , None],
        ["cur_repay_status"             , "map(PD01BD03, repay_status)"             , None],
        ["cur_moi_closed"               , "mon_itvl(PD01BR01, today)"               , None],
        ["cur_doi_last_repay"           , "day_itvl(PD01BR02, today)"               , None],
        ["cur_doi_report"               , "day_itvl(PD01BR03, today)"               , None],
        # 月度表现
        ["monthly_acc_status"           , "map(PD01CD01, d1_acc_status)"            , "acc_cat == 1"],
        ["monthly_acc_status"           , "map(PD01CD01, r4_acc_status)"            , "acc_cat == 2"],
        ["monthly_acc_status"           , "map(PD01CD01, r1_acc_status)"            , "acc_cat == 3"],
        ["monthly_acc_status"           , "map(PD01CD01, r23_acc_status)"           , "(acc_cat >= 4) & (acc_cat <= 5)"],
        ["monthly_lvl5_status"          , "map(PD01CD02, lvl5_status)"              , None],
        ["monthly_doi_last_repay"       , "day_itvl(PD01CR03, today)"               , None],
        ["monthly_doi_report"           , "day_itvl(PD01CR01, today)"               , None],
    ],
    "pboc_credit_info": [
        ["credit_org_cat"               , "map(PD02AD01, org_cat)"                  , None],
        ["credit_moi_start"             , "mon_itvl(PD02AR01, today)"               , None],
        ["credit_moi_end"               , "mon_itvl(PD02AR02, today)"               , None],
    ],
    "pboc_rel_info": [
        ["rel_org_cat"                  , "map(PD03AD01, org_cat)"                  , None],
        ["rel_biz_cat"                  , "map(PD03AD02, biz_cat)"                  , None],
        ["rel_lvl5_status"              , "map(PD03AD05, lvl5_status)"              , None],
        ["rel_repay_status"             , "map(PD03AD07, repay_status)"             , None],
        ["rel_moi_start"                , "mon_itvl(PD03AR01, today)"               , None],
        ["rel_moi_end"                  , "mon_itvl(PD03AR02, today)"               , None],
    ],
    "pboc_postfee_info": [
        ["postfee_acc_cat"              , "map(PE01AD01, postfee_acc_cat_M)"        , None],
        ["postfee_acc_status"           , "map(PE01AD03, postfee_acc_status_M)"     , None],
    ],
    "pboc_inq_rec": [
        ["inq_rec_org_cat"              , "map(PH010D01, org_cat)"                  , None],
        ["inq_rec_reason_cat"           , "map(PH010Q03, inq_reason_cat)"           , None],
    ],
}


# %%
LV2_AGG_CONF = {
    "acc_repay_60m": {
        "part": "acc_repay_60m",
        "from_": "pboc_acc_repay_60_monthly",
        "level": 1,
        "join_key": None,
        "group_key": None,
        "key_fmt": "acc_repay_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"acc_repay_moi >= {moi}", f"近{moi}月")
                      for moi in [3, 6, 9, 12, 24, 36, 48]]
                     + [("all", None, "历史"), ]),
            "status": [(f"eq{rs}", f"acc_repay_status == {rs}", f"还款状态为{rs}")
                       for rs in [1, 2, 3, 4, 5, 6, 7]],
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "期数"),
            "status_max": ("status_max", "max(acc_repay_status)", "最大还款状态"),
            "status_sum": ("status_sum", "sum(acc_repay_status)", "还款状态之和"),
            "ovd_max": ("ovd_max", "max(PD01EJ01)", "最大逾期（透支）金额"),
            "ovd_sum": ("ovd_sum", "sum(PD01EJ01)", "逾期（透支）金额之和"),
        },
        "cros": [
            (["cnt",]                           , ["mois", "status"]),
            (["status_max", "status_sum"]       , ["mois", ]),
            (["ovd_max", "ovd_sum"]             , ["mois", "status"]),
            (["ovd_max", "ovd_sum"]             , ["mois", ]),
        ],
    },
    "acc_special_trans": {
        "part": "acc_special_trans",
        "from_": "pboc_acc_special_trans",
        "level": 1,
        "join_key": None,
        "group_key": None,
        "key_fmt": "acc_special_trans_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"acc_special_trans_moi >= {moi}", f"近{moi}月")
                      for moi in [12, 24, 36, 48]]
                     + [("all", None, "历史"), ]),
            "trans": [("nor", "acc_special_trans_type <= 1", "正常类"),
                      ("ovd", "acc_special_trans_type >= 2", "逾期类")]
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "特殊交易记录数"),
            "sum": ("sum", "sum(PD01FJ01)", "特殊交易额度和"),
            "max": ("max", "max(PD01FJ01)", "特殊交易额度最大值"),
        },
        "cros": [
            (["cnt", "sum", "max"]              , ["mois", "trans"]),
        ],
    },
    "acc_special_accd": {
        "part": "acc_special_accd",
        "from_": "pboc_acc_special_accd",
        "level": 1,
        "join_key": None,
        "group_key": None,
        "key_fmt": "acc_special_accd_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"acc_special_accd_moi >= {moi}",
                       f"近{moi}月")
                      for moi in [12, 24, 36, 48]]
                     + [("all", None, "历史"), ]),
            "trans": [("nor", "acc_speical_trans_type <= 1", "正常类"),
                      ("ovd", "acc_special_trans_type >= 2", "逾期类")]
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "特殊事件记录数"),
        },
        "cros": [
            (["cnt", ]                          , ["mois", ]),
        ],
    },
    "acc_special_insts": {
        "part": "acc_special_insts",
        "from_": "pboc_acc_special_insts",
        "level": 1,
        "join_key": None,
        "group_key": None,
        "key_fmt": "acc_special_insts_{}_{}",
        "conds": {
            "mois_start": ([(f"last_{moi}m",
                             f"acc_special_insts_moi_start >= -{moi}",
                             f"近{moi}月开始") for moi in [6, 12, 24, 36, 48]]
                           + [("his", None, "历史"), ]),
            "mois_end": ([(f"folw_{moi}m",
                           f"acc_special_insts_moi_end <= {moi} & acc_special_insts_moi_end > 0",
                           f"未来{moi}月存续") for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "acc_special_insts_moi_end <= 0", "已结清"),
                            ("open", "acc_special_insts_moi_end > 0", "存续")]),
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "大额专项分期记录数"),
            "lmt_sum": ("lmt_sum", "sum(PD01HJ01)", "大额专项分期额度和"),
            "lmt_max": ("lmt_max", "sum(PD01HJ01)", "大额专项分期额度最大值"),
            "usd_sum": ("usd_sum", "sum(PD01HJ02)", "大额专项分期已用额度和"),
            "usd_max": ("usd_max", "max(PD01HJ02)", "大额专项分期已用额度最大值"),
            "usd_sum_ppor": ("usd_sum_ppor", "sum(PD01HJ02) / (sum(PD01HJ01) + 1)", "大额专项分期已用额度占比")
        },
        "cros": [
            (["cnt", "lmt_sum", "lmt_max",
              "usd_sum", "usd_max", "usd_sum_ppor"],
             ["mois_start", ]),
            (["cnt", "lmt_sum", "lmt_max",
              "usd_sum", "usd_max", "usd_sum_ppor"],
             ["mois_end", ]),
        ],
    },
}


# %%
LV1_AGG_CONF = {
    "pinfo_mobile": {
        "part": "pinfo_mobile",
        "from_": "pboc_mobile",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "pinfo_mobile_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PB01BR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("all", None, "历史"), ])
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "手机号数量"),
        },
        "cros": [
            (["cnt", ]                          , ["mois", ]),
        ]
    },
    "pinfo_res": {
        "part": "pinfo_res",
        "from_": "pboc_address",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "pinfo_res_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PB030R01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("all", None, "历史"), ]),
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "居住地数量"),
        },
        "cros": [
            (["cnt", ]                          , ["mois", ]),
        ]
    },
    "pinfo_comp": {
        "part": "pinfo_comp",
        "from_": "pboc_company",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "pinfo_comp_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PB040R01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("all", None, "历史"), ]),
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "工作单位数量"),
        },
        "cros": [
            (["cnt", ]                          , ["mois", ]),
        ]
    },
    "acc_no_cat_info": {
        "part": "acc_no_cat_info",
        "from_": "pboc_acc_info",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "acc_{}_{}",
        "conds": {
            "orgs": [
                ("org_bank", "acc_org_cat < 20", "银行账户"),
                ("org_nbank", "(acc_org_cat > 20) & (acc_org_cat < 60)", "非银机构账户"),
                ("org_other", "acc_org_cat > 90", "其他机构账户")
            ],
            "biz_cat": [
                ("biz_cat_loan", "acc_biz_cat < 200", "贷款业务"),
                ("biz_cat_card", "(acc_biz_cat > 200) & (acc_biz_cat < 300)", "信用卡业务"),
                ("biz_cat_housing", "acc_biz_cat < 120", "房贷业务"),
                ("biz_cat_auto", "(acc_biz_cat > 120) & (acc_biz_cat < 130)", "车贷业务"),
                ("biz_cat_edu", "(acc_biz_cat > 130) & (acc_biz_cat < 140)", "教育贷款业务"),
                ("biz_cat_biz", "(acc_biz_cat > 140) & (acc_biz_cat < 150)", "经营贷、农业贷业务"),
                ("biz_cat_comsu", "(acc_biz_cat > 150) & (acc_biz_cat < 160)", "消费贷业务"),
                ("biz_cat_security", "(acc_biz_cat > 300) & (acc_biz_cat < 400)", "证券融资业务"),
                ("biz_cat_leasing", "(acc_biz_cat > 400) & (acc_biz_cat < 500)", "融资租赁业务"),
                ("biz_cat_dum", "acc_biz_cat > 500", "资产处置、垫款业务")
            ],
            "mois_start": ([(f"last_{moi}m", f"acc_moi_start >= -{moi}", f"近{moi}月开立")
                            for moi in [6, 12, 24, 36, 48]]
                           + [("his", None, "历史"), ]),
            "mois_end": ([(f"folw_{moi}m", f"(acc_moi_end <= {moi}) & (acc_moi_end > 0)", f"未来{moi}月存续")
                          for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "acc_moi_end <= 0", "已结清"),
                            ("open", "acc_moi_end > 0", "存续")]),
            "cur_acc_status": [
                ("cur_acc_closed", "cur_acc_status == 0", "当前关闭、未激活"),
                ("cur_acc_nor", "cur_acc_status == 11", "当前正常"),
                ("cur_acc_ovd", "(cur_acc_status > 20) & (cur_acc_status < 30)", "当前逾期"),
                ("cur_acc_abnor", "(cur_acc_status > 30) & (cur_acc_status < 40)", "当前异常"),
                ("cur_acc_dum", "cur_acc_status == 99", "当前呆账"),
            ],
            "cur_repay_status": [(f"eq{rs}", f"acc_status == {rs}", f"最近月度为{rs}")
                                 for rs in [1, 2, 3, 4, 5, 6, 7]],
            "monthly_acc_status": [
                ("monthly_acc_closed", "monthly_acc_status == 0", "月度关闭"),
                ("monthly_acc_nor", "monthly_acc_status == 11", "月度正常"),
                ("monthly_acc_ovd", "(monthly_acc_status > 20) & (monthly_acc_status < 30)", "月度逾期"),
                ("monthly_acc_abnor", "(monthly_acc_status > 30) & (monthly_acc_status < 40)", "月度异常"),
                ("monthly_acc_dum", "monthly_acc_status == 99", "月度呆账"),
            ],
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "账户数"),
            "ltm_sum": ("lmt_sum", "sum(acc_lmt)", "借款、授信额度之和"),
            "ltm_max": ("lmt_max", "max(acc_lmt)", "借款、授信额度最大值"),
        },
        "cros": [
            (["cnt", "ltm_sum", "ltm_max"]      , ["orgs", "mois_start"]),
            (["cnt", "ltm_sum", "ltm_max"]      , ["orgs", "mois_end"]),
            (["cnt", "ltm_sum", "ltm_max"]      , ["biz_cat", "mois_start"]),
            (["cnt", "ltm_sum", "ltm_max"]      , ["biz_cat", "mois_end"]),
            (["cnt", "ltm_sum", "ltm_max"]      , ["biz_cat", "cur_acc_status"]),
            (["cnt", "ltm_sum", "ltm_max"]      , ["biz_cat", "monthly_acc_status"]),
        ],
    },
    "acc_cat_info": {
        "part": "acc_cat_info",
        "from_": "pboc_acc_info",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "acc_{}_{}",
        "conds": {
            "acc_cat": {
                "c1": ("c1"     , "acc_cat == 99"       , "c1"),
                "d1": ("d1"     , "acc_cat == 1"        , "d1"),
                "r4": ("r4"     , "acc_cat == 2"        , "r4"),
                "r1": ("r1"     , "acc_cat == 3"        , "r1"),
                "r2": ("r2"     , "acc_cat == 4"        , "r2"),
                "r3": ("r3"     , "acc_cat == 5"        , "r3"),
                "d1r4": ("d1r4" , "acc_cat <= 2"        , "d1r4"),
                "d1r41": ("d1r41"   , "acc_cat <= 3"    , "d1r41"),
                "r23": ("r23"   , "(acc_cat >= 4) & (acc_cat <= 5)" , "r23"),
            },
            "cur_lvl5_status": [
                ("cur_lvl5_nor", "cur_lvl5_status == 1", "最新五级分类正常",),
                ("cur_lvl5_ovd", "cur_lvl5_status >= 2", "最新五级分类逾期",),
            ],
            "monthly_lvl5_status": [
                ("monthly_lvl5_nor", "monthly_lvl5_status == 1", "月度五级分类正常",),
                ("monthly_lvl5_ovd", "monthly_lvl5_status >= 2", "月度五级分类逾期",),
            ],
            "trans_status": [(f"trans_status_eq{ts}", f"acc_trans_status == {ts}",
                              f"转移时状态为{ts}")
                             for ts in [0, 1, 2, 3, 4, 5, 6, 7]],
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "账户数"),
            "lmt_sum": ("lmt_sum", "sum(acc_lmt)", "借款、授信额度之和"),
            "lmt_max": ("lmt_max", "max(acc_lmt)", "借款、授信额度最大值"),
            "folw_prd_max": ("folw_prd_max", "max(acc_moi_end)", "账户剩余最长期限"),
            "last_prd_max": ("last_prd_max", "max(-acc_moi_start)", "首个账户距今"),
            # 最新表现
            "cur_ots_sum": ("cur_ots_sum", "sum(PD01BJ01)", "最新余额之和"),
            "cur_ots_max": ("cur_ots_max", "max(PD01BJ01)", "最新余额最大值"),
            "last_repay_mcnt": ("last_repay_mcnt", "min(mon_itvl(today, PD01BR02))", "最近还款距今"),
            # TODO: implement `argmin`
            # "last_repay_amt": ("last_repay_amt", "PD01BJ02[argmin(mon_itvl(today, PD01BR02))]", "最近还款金额"),
            # "last_repay_amt": ("last_repay_amt", "PD01BJ02[argmin(mon_itvl(today, PD01BR02))]", "最近还款金额"),
            # 月度表现
            "monthly_ots_sum": ("monthly_ots_sum", "sum(PD01CJ01)", "月度余额之和"),
            "monthly_ots_max": ("monthly_ots_max", "max(PD01CJ01)", "月度余额最大值"),
            "monthly_usd_sum": ("monthly_usd_sum", "sum(PD01CJ02)", "月度已用额度之和"),
            "monthly_usd_max": ("monthly_usd_max", "max(PD01CJ02)", "月度已用额度最大值"),
            "monthly_special_insts_sum": ("monthly_special_insts_sum", "sum(PD01CJ03)", "月度未出单大额专项余额之和"),
            "monthly_special_insts_max": ("monthly_special_insts_max", "max(PD01CJ03)", "月度未出单大额专项余额最大值"),
            "duepay_sum": ("duepay_sum", "sum(PD01CJ04)", "本月应还之和"),
            "duepay_max": ("duepay_max", "max(PD01CJ04)", "本月应还最大值"),
            "ovd_sum": ("ovd_sum", "sum(PD01CJ06)", "逾期总额之和"),
            "ovd_max": ("ovd_max", "max(PD01CJ06)", "逾期总额最大值"),
            "m2_ovd_pri_sum": ("m2_ovd_pri_sum", "sum(PD01CJ07)", "m2逾期本金之和"),
            "m2_ovd_pri_max": ("m2_ovd_pri_max", "max(PD01CJ07)", "m2逾期本金最大值"),
            "m3_ovd_pri_sum": ("m3_ovd_pri_sum", "sum(PD01CJ08)", "m3逾期本金之和"),
            "m3_ovd_pri_max": ("m3_ovd_pri_max", "max(PD01CJ08)", "m3逾期本金最大值"),
            "m46_ovd_pri_sum": ("m46_ovd_pri_sum", "sum(PD01CJ09)", "m46逾期本金之和"),
            "m46_ovd_pri_max": ("m46_ovd_pri_max", "max(PD01CJ09)", "m46逾期本金最大值"),
            "m7p_ovd_pri_sum": ("m7p_ovd_pri_sum", "sum(PD01CJ10)", "m7p逾期本金之和"),
            "m7p_ovd_pri_max": ("m7p_ovd_pri_max", "max(PD01CJ10)", "m7p逾期本金最大值"),
            "m7p_ovd_sum": ("m7p_ovd_sum", "sum(PD01CJ11)", "m7p透支未付余额之和"),
            "m7p_ovd_max": ("m7p_ovd_max", "sum(PD01CJ11)", "m7p透支未付余额最大值"),
            "last_6m_avg_usd_sum": ("last_6m_avg_usd_sum", "sum(last_6m_avg_usd)", "最近6个月平均使用额度之和"),
            "last_6m_avg_usd_max": ("last_6m_avg_usd_max", "sum(last_6m_avg_usd)", "最近6个月平均使用额度最大值"),
            "last_6m_max_usd_max": ("last_6m_max_usd_max", "sum(last_6m_max_usd)", "最近6个月最大使用额度最大值"),
            "monthly_folw_prd_max": ("monthly_folw_prd_max", "max(PD01CS01)", "剩余还款期数最大值"),
            "monthly_ovd_prd_sum": ("monthly_ovd_prd_sum", "sum(PD01CS02)", "当前逾期期数之和"),
            "monthly_ovd_prd_max": ("monthly_ovd_prd_max", "max(PD01CS02)", "当前逾期期数最大值"),
        },
        "cros": [
            (["cnt", ]              , ["cur_lvl5_status", "acc_cat.[d1r41]"]),
            (["cnt", ]              , ["monthly_lvl5_status", "acc_cat.[d1r41]"]),
            (["cnt", ]              , ["trans_status", "acc_cat.[c1]"]),
            (["lmt_sum", "lmt_max", "last_prd_max"],
             [("acc_cat", "c1", "d1", "r4", "r1", "r2", "r3")]),
            (["folw_prd_max",],
             [("acc_cat", "d1r4", "r1")]),
            (["cur_ots_sum", "cur_ots_max", "monthly_ots_sum", "monthly_ots_max"],
             [("acc_cat", "c1", "d1", "r4", "r1", "r23")]),
            (["last_repay_mcnt",],
             [("acc_cat", "c1", "d1r41", "r23")]),
            # (["last_repay_amt",],
            #  [("acc_cat", "d1r41", "r23")]),
            (["monthly_usd_sum", "monthly_usd_max", "monthly_special_insts_sum",
             "monthly_special_insts_max",],
             [("acc_cat", "r2", )]),
            (["duepay_sum", "duepay_max", "ovd_sum", "ovd_max"],
             [("acc_cat", "d1r41", "r2")]),
            (["m2_ovd_pri_sum", "m2_ovd_pri_max",
              "m3_ovd_pri_sum", "m3_ovd_pri_max",
              "m46_ovd_pri_sum", "m46_ovd_pri_max",
              "m7p_ovd_pri_sum", "m7p_ovd_pri_max"],
             [("acc_cat", "d1r41", )]),
            (["m7p_ovd_sum", "m7p_ovd_max"],
             [("acc_cat", "r3", )]),
            (["last_6m_avg_usd_sum", "last_6m_avg_usd_max", "last_6m_max_usd_max"],
             [("acc_cat", "r23")]),
            (["monthly_folw_prd_max", "monthly_ovd_prd_sum", "monthly_ovd_prd_max"],
             [("acc_cat", "d1r41", "r2")]),
        ],
    },
    "credit_info": {
        "part": "credit_info",
        "from_": "pboc_credit_info",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "credit_{}_{}",
        "conds": {
            "orgs": [
                ("org_bank", "credit_org_cat < 20", "银行账户"),
                ("org_nbank", "(credit_org_cat > 20) & (credit_org_cat < 60)", "非银机构账户"),
                ("org_other", "credit_org_cat > 90", "其他机构账户")
            ],
            "mois_start": ([(f"last_{moi}m", f"credit_moi_start >= -{moi}", f"近{moi}月开始")
                            for moi in [6, 12, 24, 36, 48]]
                           + [("his", None, "历史"), ]),
            "mois_end": ([(f"folw_{moi}m", f"(credit_moi_end <= {moi}) & (credit_moi_end > 0)", f"未来{moi}月存续")
                          for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "credit_moi_end <= 0", "已结束"),
                            ("open", "credit_moi_end > 0", "存续")]),
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "授信数量"),
            "lmt_sum": ("lmt_sum", "sum(PD02AJ01)", "授信额度之和"),
            "lmt_max": ("lmt_max", "max(PD02AJ01)", "授信额度最大值"),
            "lmt2_sum": ("lmt2_sum", "sum(PD02AJ03)", "授信限额之和"),
            "lmt2_max": ("lmt2_max", "max(PD02AJ03)", "授信限额最大值"),
            "usd_sum": ("usd_sum", "sum(PD02AJ04)", "已用额度之和"),
            "usd_max": ("usd_max", "max(PD02AJ04)", "已用额度最大值"),
        },
        "cros": [
            [["cnt", "lmt_sum", "lmt_max", "lmt2_sum", "lmt2_max",
              "usd_sum", "usd_max"],
             ["mois_start", "orgs"]],
            [["cnt", "lmt_sum", "lmt_max", "lmt2_sum", "lmt2_max",
              "usd_sum", "usd_max"],
             ["mois_end", "orgs"]],
        ]
    },
    "rel_info": {
        "from_": "pboc_rel_info",
        "part": "rel_info",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "rel_{}_{}",
        "conds": {
            "orgs": [
                ("org_bank", "rel_org_cat < 20", "银行账户"),
                ("org_nbank", "(rel_org_cat > 20) & (rel_org_cat < 60)", "非银机构账户"),
                ("org_other", "rel_org_cat > 90", "其他机构账户")
            ],
            "biz_cat": [
                ("biz_cat_loan", "rel_biz_cat < 200", "贷款业务"),
                ("biz_cat_card", "(rel_biz_cat > 200) & (rel_biz_cat < 300)", "信用卡业务"),
                ("biz_cat_housing", "rel_biz_cat < 120", "房贷业务"),
                ("biz_cat_auto", "(rel_biz_cat > 120) & (rel_biz_cat < 130)", "车贷业务"),
                ("biz_cat_edu", "(rel_biz_cat > 130) & (rel_biz_cat < 140)", "教育贷款业务"),
                ("biz_cat_biz", "(rel_biz_cat > 140) & (rel_biz_cat < 150)", "经营贷、农业贷业务"),
                ("biz_cat_comsu", "(rel_biz_cat > 150) & (rel_biz_cat < 160)", "消费贷业务"),
                ("biz_cat_security", "(rel_biz_cat > 300) & (rel_biz_cat < 400)", "证券融资业务"),
                ("biz_cat_leasing", "(rel_biz_cat > 400) & (rel_biz_cat < 500)", "融资租赁业务"),
                ("biz_cat_dum", "rel_biz_cat > 500", "资产处置、垫款业务")
            ],
            "lvl5_status": [
                ("lvl5_nor", "rel_lvl5_status == 1", "五级分类正常",),
                ("lvl5_ovd", "rel_lvl5_status >= 2", "五级分类逾期",),
            ],
            "repay_status": [(f"eq{rs}", f"rel_repay_status == {rs}", f"最近月度为{rs}")
                             for rs in [1, 2, 3, 4, 5, 6, 7]],
            "mois_start": ([(f"last_{moi}m", f"rel_moi_start >= -{moi}", f"近{moi}月开始")
                            for moi in [6, 12, 24, 36, 48]]
                           + [("his", None, "历史"), ]),
            "mois_end": ([(f"folw_{moi}m", f"(rel_moi_end <= {moi}) & (rel_moi_end > 0)",
                           f"未来{moi}月存续")
                          for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "rel_moi_end <= 0", "已结束"),
                            ("open", "rel_moi_end > 0", "存续")]),
            "ovd_month": ([(f"ovd_prd_eq{om}m", f"PD03AS01 == {om}", f"逾期月数为{om}")
                           for om in [1, 2, 3, 4, 5, 6]]
                          + [("ovd_prd_ge7m", "PD03AS01 >= 7", "逾期月数大于等于7")])
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "相关还款责任数"),
            "respon_sum": ("resp_sum", "sum(PD03AJ01)", "责任金额之和"),
            "respon_max": ("resp_max", "max(PD03AJ01)", "责任金额最大值"),
            "acc_sum": ("acc_sum", "sum(PD03AJ02)", "账户金额之和"),
            "acc_max": ("acc_max", "max(PD03AJ02)", "账户金额最大值"),
            "repay_status_max": ("repay_status_max", "max(rel_repay_status)", "逾期月数最大值"),
            "repay_status_sum": ("repay_status_sum", "sum(rel_repay_status)", "逾期月数之和"),
            "ovd_month_max": ("ovd_month_max", "max(PD03AS01)", "逾期月数最大值"),
            "ovd_month_sum": ("ovd_month_sum", "sum(PD03AS01)", "逾期月数之和"),
            "folw_prd_max": ("folw_prd_max", "max(rel_moi_end)", "剩余最长期限"),
            "last_prd_max": ("last_prd_max", "max(-rel_moi_start)", "首个账户距今"),
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
    },
    "postfee_info": {
        "part": "postfee_info",
        "from_": "pboc_postfee_info",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "postfee_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PE01AR02, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "acc_cat": [
                ("tel", "postfee_acc_cat == 1", "电信业务"),
                ("pub", "postfee_acc_cat == 2", "水电费等公共事业"),
            ],
            "acc_status": {
                "nor": ("nor", "postfee_acc_status == 0", "正常"),
                "ovd": ("ovd", "postfee_acc_status == 1", "欠费"),
            },
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "账户数"),
            "ovd_sum": ("ovd_sum", "sum(PE01AJ01)", "欠费金额之和"),
            "ovd_max": ("ovd_max", "max(PE01AJ01)", "欠费金额最大值"),
        },
        "cros": [
            (["cnt", "ovd_sum", "ovd_max"] , ["acc_cat", "mois"]),
        ],
    },
    "taxs": {
        "part": "taxs",
        "from_": "pboc_taxs",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "tax_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF01AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "账户数"),
            "ovd_sum": ("ovd_sum", "sum(PF01AJ01)", "欠税额之和"),
            "ovd_max": ("ovd_max", "max(PF01AJ01)", "欠税额最大值"),
        },
        "cros": [
            (["cnt", "ovd_sum", "ovd_max"]  , ["mois", ]),
        ],
    },
    "lawsuit": {
        "part": "lawsuit",
        "from_": "pboc_lawsuit",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "lawsuit_{}_{}",
        "conds":{
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF02AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "None": [("all", None, "所有记录中"), ]
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "民事判决案件数"),
            "target_sum": ("target_sum", "sum(PF02AJ01)", "民事判决标的金额之和"),
            "target_max": ("target_max", "max(PF02AJ01)", "民事判决标的金额最大值"),
            "prd_max": ("prd_max", "max(mon_itvl(today, PF02AR01))", "最早民事判决距今"),
            "prd_min": ("prd_min", "min(mon_itvl(today, PF02AR01))", "最晚民事判决距今"),
        },
        "cros": [
            (["cnt", "target_sum", "target_max"]    , ["mois"]),
            (["prd_max", "prd_min"]                 , ["None"]),
        ],
    },
    "enforcement": {
        "part": "enforcement",
        "from_": "pboc_enforcement",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "enforcement_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF03AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "None": [("all", None, "所有记录中"), ]
        },
        "aggs": {
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
            (["prd_max", "prd_min"]                 , ["None"]),
        ],
    },
    "gov_punishment": {
        "part": "gov_punishment",
        "from_": "pboc_gov_punishment",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "gov_punishment_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF04AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "None": [("all", None, "所有记录中"), ]

        },
        "aggs": {
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
    },
    "housing_fund": {
        "part": "housing_fund",
        "from_": "pboc_housing_fund",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "housing_fund_{}_{}",
        "conds": {
            "mois_start": ([(f"last_{moi}m", f"mon_itvl(PF05AR02, today) >= -{moi}",
                             f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                           + [("his", None, "历史"), ]),
            "mois_end": ([(f"last_{moi}m",
                           f"(mon_itvl(PF05AR03, today) <= {moi}) & (mon_itvl(PF05AR03, today) > 0)",
                           f"未来{moi}月存续") for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "mon_itvl(PF05AR03, today) <= 0", "已结束"),
                            ("open", "mon_itvl(PF05AR03, today) > 0", "存续")]),
            "None": [("all", None, "所有记录中"), ]
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "住房公积金账户数量"),
            "sum": ("sum", "sum(PF05AJ01)", "住房公积金月缴存额之和"),
            "max": ("max", "max(PF05AJ01)", "住房公积金月缴存额最大值"),
            "fstpay_prd_max": ("fstpay_prd_max", "max(mon_itvl(today, PF05AR02))", "最早初缴距今"),
            "fstpay_prd_min": ("fstpay_prd_min", "min(mon_itvl(today, PF05AR02))", "最晚初缴距今"),
            "lastpay_prd_max": ("prd_max", "max(mon_itvl(today, PF05AR03))", "最早缴至距今"),
            "lastpay_prd_min": ("prd_min", "min(mon_itvl(today, PF05AR03))", "最晚缴至距今"),
        },
        "cros": [
            (["cnt", "sum", "max"]                , ["None"]),
            # (["cnt", "sum", "max"]              , ["mois_start"]),
            # (["cnt", "sum", "max"]              , ["mois_end"]),
            # (["fstpay_prd_max", "fstpay_prd_min",
            #   "fstpay_prd_max", "fstpay_prd_min"],
            #  ["None"])
        ],
    },
    "allowance": {
        "part": "sub_allowance",
        "from_": "pboc_sub_allowance",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "allowance_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF06AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [12, 24, 36, 48]]
                     + [("his", None, "历史"), ])
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "低保救助记录数量"),
        },
        "cros": [
            (["cnt",]                       , ["mois",]),
        ],
    },
    "pro_cert": {
        "part": "pro_cert",
        "from_": "pboc_pro_cert",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "cert_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF07AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "None": [("all", None, "所有记录中"), ]
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "职业资格数量"),
        },
        "cros": [
            (["cnt",]                       , ["mois",]),
        ],
    },
    "gov_award": {
        "part": "gov_award",
        "from_": "pboc_gov_award",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "award_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF08AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "None": [("all", None, "所有记录中"), ]
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "奖励数量"),
        },
        "cros": [
            (["cnt",]                       , ["None",]),
        ],
    },
    "inq_rec": {
        "part": "inq_rec",
        "from_": "pboc_inq_rec",
        "level": 0,
        "join_key": None,
        "group_key": None,
        "key_fmt": "inq_rec_{}_{}",
        "conds": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PH010R01, today) >= -{moi}",
                       f"近{moi}月") for moi in [12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "orgs": [
                ("org_bank", "inq_rec_org_cat < 20", "银行账户"),
                ("org_nbank", "(inq_rec_org_cat > 20) & (inq_rec_org_cat < 60)", "非银机构账户"),
                ("org_other", "inq_rec_org_cat > 90", "其他机构账户")
            ],
            "inq_reason": [
                ("for_pre", "inq_rec_reason_cat < 20", "贷前审批"),
                ("for_after", "(inq_rec_reason_cat > 20) & (inq_rec_reason_cat < 30)", "贷后管理"),
                ("for_rel", "(inq_rec_reason_cat > 30) & (inq_rec_reason_cat < 40)", "关联审查"),
                ("for_others", "inq_rec_reason_cat > 40", "其他原因审查")
            ],
        },
        "aggs": {
            "cnt": ("cnt", "count(_)", "查询数量"),
        },
        "cros": [
            [["cnt",]                       , ["mois", "orgs", "inq_reason"]],
        ],
    },
}


