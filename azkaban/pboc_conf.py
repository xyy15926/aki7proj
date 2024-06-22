#!/usr/bin/env python3
# ----------------------------------------------------------
#   Name: pboc_conf.py
#   Author: xyy15926
#   Created: 2022-11-10 21:44:59
#   Updated: 2024-06-21 16:29:53
#   Description:
# ----------------------------------------------------------

# %%
MAPPERS = {
    "repay_status": {
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
    },
    "repay_status_spec": {
        "D": (31 , "担保人代还"),
        "Z": (32 , "以资抵债"),
        "B": (71 , "呆账"),
    },
    "currency": {
        "USD": (2       , "USD"),
        "EUR": (3       , "EUR"),
        "JPY": (4       , "JPY"),
        "CNY": (1       , "CNY"),
        "AUD": (5       , "AUD"),
        "RUB": (6       , "RUB"),
        "CAD": (7       , "CAD"),
    },
    "exchange_rate": {
        "USD": (7       , "USD"),
        "EUR": (7.7     , "EUR"),
        "JPY": (0.05    , "JPY"),
        "CNY": (1       , "CNY"),
        "AUD": (4.7     , "AUD"),
        "RUB": (0.07    , "RUB"),
        "CAD": (5.3     , "CAD"),
    },
    "special_trans_type": {
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
    },
    "biz_cat": {
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
    },
    "cdr_cat": {
        "D1": (1        , "非循环贷账户"),
        "R1": (3        , "循环贷账户"),
        "R2": (4        , "贷记卡账户"),
        "R3": (5        , "准贷记卡账户"),
        "R4": (2        , "循环额度下分账户"),
        "C1": (99       , "催收账户"),
    },
    "org_cat": {
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

    },
    "trans_status": {
        "0": (0         , "债务人即将违约时自动垫款"),
        "1": (1         , "逾期1-30天"),
        "2": (2         , "逾期31-60天"),
        "3": (3         , "逾期61-90天"),
        "4": (4         , "逾期91-120天"),
        "5": (5         , "逾期121-150天"),
        "6": (6         , "逾期151-180天"),
        "7": (7         , "逾期180天以上"),
        "9": (0         , "未知"),
    },
    # 0: 结清、关闭、未激活
    # 1x: 正常
    # 2x: 逾期、催收、追偿
    # 3x: 平仓、止付、冻结、转出
    # 99: 呆账
    "c1_acc_status": {
        "1": (22        , "催收"),
        "2": (0         , "结束"),
    },
    "d1_acc_status": {
        "1": (11        , "正常"),
        "2": (21        , "逾期"),
        "3": (0         , "结清"),
        "4": (99        , "呆账"),
        "5": (2         , "转出"),
        "6": (31        , "担保物不足"),
        "7": (32        , "强制平仓"),
        "8": (23        , "司法追偿"),
    },
    "r4_acc_status": {
        "1": (11        , "正常"),
        "2": (21        , "逾期"),
        "3": (0         , "结清"),
        "4": (99        , "呆账"),
        "6": (31        , "担保物不足"),
        "8": (23        , "司法追偿"),
    },
    "r1_acc_status": {
        "1": (11        , "正常"),
        "2": (21        , "逾期"),
        "3": (0         , "结清"),
        "4": (99        , "呆账"),
        "5": (33        , "银行止付"),
        "6": (31        , "担保物不足"),
        "8": (23        , "司法追偿"),
    },
    "r23_acc_status": {
        "1": (11        , "正常"),
        "2": (35        , "冻结"),
        "3": (34        , "止付"),
        "31":(33        , "银行止付"),
        "4": (0         , "销户"),
        "5": (99        , "呆账"),
        "6": (1         , "未激活"),
        "8": (23        , "司法追偿"),
    },
    "lvl5_status": {
        "1": (1         , "正常"),
        "2": (2         , "关注"),
        "3": (3         , "次级"),
        "4": (4         , "可疑"),
        "5": (5         , "损失"),
        "6": (6         , "违约"),
        "9": (1         , "未分类"),
    },
    "credit_lmt_cat": {
        "10": (10       , "循环贷款额度"),
        "20": (20       , "非循环贷款额度"),
        "30": (30       , "信用卡共享额度"),
        "31": (31       , "信用卡独立额度"),
    },
    "credit_protocal_status": {
        "1": (1         , "有效"),
        "2": (2         , "到期/失效"),
    },
    "postfee_acc_cat_M": {
        "TE": (1        , "电信缴费账户"),
        "UE": (2        , "公用事业缴费账户"),
    },
    "postfee_acc_status_M": {
        "0": (0         , "正常"),
        "1": (1         , "欠费"),
    },
    "inq_reason_cat": {
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
    },
    "repay_freq": {
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
    },
    "edu_record": {
        "10": (10       , "研究生"),
        "20": (20       , "本科"),
        "30": (30       , "大专"),
        "40": (40       , "中专、职高、技校"),
        "60": (60       , "高中"),
        "90": (90       , "其他"),
        "91": (91       , "初中及以下"),
        "99": (99       , "未知"),
    },
    "edu_degree": {
        "1": (1         , "名誉博士"),
        "2": (2         , "博士"),
        "3": (3         , "硕士"),
        "4": (4         , "学士"),
        "5": (5         , "无"),
        "9": (9         , "未知"),
        "0": (0         , "其他"),
    },
    "emp_status": {
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
    },
    "marital_status": {
        "10": (10       , "未婚"),
        "20": (20       , "已婚"),
        "30": (30       , "丧偶"),
        "40": (40       , "离婚"),
        "91": (91       , "单身"),
        "99": (99       , "未知"),
    },
    "res_status": {
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
    },
    # 单位性质
    "comp_character": {
        "10": (10       , "机关、事业单位"),
        "20": (20       , "国有企业"),
        "30": (30       , "外资企业"),
        "40": (40       , "个体、私营企业"),
        "50": (50       , "其他（三资企业、民营企业、民间团体）"),
        "99": (99       , "未知"),
    },
    # 单位行业
    "comp_industry": {
        "A": (11        , "农、林、牧、渔业"),
        "B": (21        , "采矿业"),
        "C": (22        , "制造业"),
        "D": (23        , "电力、热力、燃气及水生产和供应业"),
        "E": (24        , "建筑业"),
        "F": (31        , "批发和零售业"),
        "G": (32        , "交通运输、仓储和邮储业"),
        "H": (33        , "住宿和餐饮业"),
        "1": (34        , "信息传输、软件和信息技术服务业"),
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
    },
    # 职业
    "comp_profession": {
        "0": (11        , "国家机关、党群组织、企业、事业单位负责人"),
        "1": (31        , "专业技术人员"),
        "3": (41        , "办事人员和有关人员"),
        "4": (42        , "商业、服务业人员"),
        "5": (21        , "农、林、牧、渔、水利业生产人员"),
        "6": (32        , "生产、运输设备操作人员及有关人员"),
        "X": (12        , "军人"),
        "Y": (98        , "不便分类的其他从业人员"),
        "Z": (99        , "未知"),
    },
    # 职务
    "comp_position": {
        "1": (1         , "高级领导"),
        "2": (2         , "中级领导"),
        "3": (3         , "一般员工"),
        "4": (98        , "其他"),
        "9": (99        , "未知"),
    },
    # 职称
    "comp_prof_title": {
        "0": (98        , "无"),
        "1": (1         , "高级"),
        "2": (2         , "中级"),
        "3": (3         , "初级"),
        "9": (99        , "未知"),
    },
    "housing_fund_status": {
        "1": (1         , "缴交"),
        "2": (2         , "封存"),
        "3": (3         , "销户"),
    },
}


# %%
TRANS_CONF = {
    "pboc_acc_repay_60_monthly": [
        ["acc_repay_status"             , "map(PD01ED01, repay_status)"         , None  , "还款状态"],
        ["acc_repay_status_spec"        , "map(PD01ED01, repay_status_spec)"    , None  , "还款状态"],
        ["acc_repay_moi"                , "mon_itvl(PD01ER03, today)"           , None  , "还款月距今月"]
    ],
    "pboc_acc_special_trans": [
        ["acc_special_trans_type"       , "map(PD01FD01, special_trans_type)"   , None  , "特殊交易类型"],
        ["acc_special_trans_moi"        , "mon_itvl(PD01FR01, today)"           , None  , "特殊交易距今月"],
    ],
    "pboc_acc_special_accd": [
        ["acc_special_accd_moi"         , "mon_itvl(PD01GR01, today)"           , None  , "特殊事件距今月"],
    ],
    # PD01HJ02 无法在此层换算汇率，只能在后续中特殊处理
    "pboc_acc_special_insts": [
        ["acc_special_insts_moi_start"  , "mon_itvl(PD01HR01, today)"           , None  , "大额分期起始距今月"],
        ["acc_special_insts_moi_end"    , "mon_itvl(PD01HR02, today)"           , None  , "大额分期结束距今月"],
        ["acc_special_insts_monthly_repay"    , "PD01HJ02 / mon_itvl(PD01HR02, PD01HR01)"     , None  , "大额分期月均还款"],
    ],
    "pboc_acc_info": [
        # 账户信息
        ["acc_cat"                      , "map(PD01AD01, cdr_cat)"              , None  , "账户类型"],
        ["acc_org_cat"                  , "map(PD01AD02, org_cat)"              , None  , "管理机构类型"],
        ["acc_biz_cat"                  , "map(PD01AD03, biz_cat)"              , None  , "账户业务类型"],
        # 贷款大部分不为外币，仅双币信用卡场合同时分别对应本币、外币账户
        # 下述聚集中将币种过滤直接设置在 `acc_cat.[r2]` 中
        ["acc_currency"                 , "map(PD01AD04, currency)"             , None  , "账户币种"],
        ["acc_exchange_rate"            , "map(PD01AD04, exchange_rate, 1)"     , None  , "账户汇率"],
        ["acc_repay_freq"               , "map(PD01AD06, repay_freq)"           , None  , "D1R41账户还款频率"],
        ["acc_trans_status"             , "map(PD01AD10, trans_status)"         , None  , "C1账户转移时状态"],
        ["PD01AJ01"                     , "smul(PD01AJ01, acc_exchange_rate)"   , None  , "C1D4R1借款金额（本币）"],
        ["PD01AJ02"                     , "smul(PD01AJ02, acc_exchange_rate)"   , None  , "R123授信金额（本币）"],
        ["PD01AJ03"                     , "smul(PD01AJ03, acc_exchange_rate)"   , None  , "R23共享授信额度（本币）"],
        ["acc_lmt"                      , "cb_fst(PD01AJ01, PD01AJ02)"          , None  , "账户借款、授信额"],
        ["acc_moi_range"                , "mon_itvl(PD01AR02, PD01AR01)"        , None  , "账户预期月数"],
        ["acc_moi_start"                , "mon_itvl(PD01AR01, today)"           , None  , "账户起始距今月"],
        ["acc_doi_start"                , "day_itvl(PD01AR01, today)"           , None  , "账户起始距今日"],
        ["acc_moi_end"                  , "mon_itvl(PD01AR02, today)"           , None  , "账户（预期）结束距今月"],        # Mixed: mixed_acc_moi_folw
        # 最新表现
        ["cur_acc_status"               , "map(PD01BD01, c1_acc_status)"        , "acc_cat == 99"   , "最近状态"],          # Mixed: mixed_acc_status
        ["cur_acc_status"               , "map(PD01BD01, d1_acc_status)"        , "acc_cat == 1"    , "最近状态"],          # Mixed: mixed_acc_status
        ["cur_acc_status"               , "map(PD01BD01, r4_acc_status)"        , "acc_cat == 2"    , "最近状态"],          # Mixed: mixed_acc_status
        ["cur_acc_status"               , "map(PD01BD01, r1_acc_status)"        , "acc_cat == 3"    , "最近状态"],          # Mixed: mixed_acc_status
        ["cur_acc_status"               , "map(PD01BD01, r23_acc_status)"       , "(acc_cat >= 4) & (acc_cat <= 5)" , "最近状态"],  # Mixed: mixed_acc_status
        ["cur_lvl5_status"              , "map(PD01BD03, lvl5_status)"          , None  , "最近5级分类"],                   # Mixed: mixed_lvl5_status
        ["cur_repay_status"             , "map(PD01BD04, repay_status)"         , None  , "最近还款状态"],
        ["PD01BJ01"                     , "smul(PD01BJ01, acc_exchange_rate)"   , None  , "账户余额（本币）"],              # Mixed: mixed_ots
        ["PD01BJ02"                     , "smul(PD01BJ02, acc_exchange_rate)"   , None  , "最近还款金额（本币）"],          # Mixed: mixed_last_repay_amt
        ["cur_moi_closed"               , "mon_itvl(PD01BR01, today)"           , None  , "最近关闭时间"],                  # Mixed: mixed_acc_moi_folw
        ["cur_doi_last_repay"           , "day_itvl(PD01BR02, today)"           , None  , "最近还款距今日"],                # Mixed: mixed_doi_last_repay
        ["cur_doi_report"               , "day_itvl(PD01BR03, today)"           , None  , "最近报告日期距今日"],            # Mixed: mixed_doi_report
        # 月度表现
        ["monthly_acc_status"           , "map(PD01CD01, d1_acc_status)"        , "acc_cat == 1"    , "月度状态"],          # Mixed: mixed_acc_status
        ["monthly_acc_status"           , "map(PD01CD01, r4_acc_status)"        , "acc_cat == 2"    , "月度状态"],          # Mixed: mixed_acc_status
        ["monthly_acc_status"           , "map(PD01CD01, r1_acc_status)"        , "acc_cat == 3"    , "月度状态"],          # Mixed: mixed_acc_status
        ["monthly_acc_status"           , "map(PD01CD01, r23_acc_status)"       , "(acc_cat >= 4) & (acc_cat <= 5)" , "月度状态"],  # Mixed: mixed_acc_status
        ["monthly_lvl5_status"          , "map(PD01CD02, lvl5_status)"          , None  , "月度5级分类"],                   # Mixed: mixed_lvl5_status
        ["PD01CJ01"                     , "smul(PD01CJ01, acc_exchange_rate)"   , None  , "账户余额（本币）"],              # Mixed: mixed_ots
        ["PD01CJ02"                     , "smul(PD01CJ02, acc_exchange_rate)"   , None  , "R2已用额度（本币）"],
        ["PD01CJ03"                     , "smul(PD01CJ03, acc_exchange_rate)"   , None  , "R2未出单大额专项分期余额（本币）"],
        ["PD01CJ04"                     , "smul(PD01CJ04, acc_exchange_rate)"   , None  , "D1R412本月应还款（本币）"],
        ["PD01CJ05"                     , "smul(PD01CJ05, acc_exchange_rate)"   , None  , "本月实还款（本币）"],            # Mixed: mixed_last_repay_amt
        ["PD01CJ06"                     , "smul(PD01CJ06, acc_exchange_rate)"   , None  , "D1R412当前逾期总额（本币）"],
        ["PD01CJ07"                     , "smul(PD01CJ07, acc_exchange_rate)"   , None  , "D1R41 M2未还本金（本币）"],
        ["PD01CJ08"                     , "smul(PD01CJ08, acc_exchange_rate)"   , None  , "D1R41 M3未还本金（本币）"],
        ["PD01CJ09"                     , "smul(PD01CJ09, acc_exchange_rate)"   , None  , "D1R41 M46未还本金（本币）"],
        ["PD01CJ10"                     , "smul(PD01CJ10, acc_exchange_rate)"   , None  , "D1R41 M7p未还本金（本币）"],
        ["PD01CJ11"                     , "smul(PD01CJ11, acc_exchange_rate)"   , None  , "R3 M7p未付余额（本币）"],
        ["PD01CJ12"                     , "smul(PD01CJ12, acc_exchange_rate)"   , None  , "R2 最近6个月平均使用额度（本币）"],
        ["PD01CJ13"                     , "smul(PD01CJ13, acc_exchange_rate)"   , None  , "R3 最近6个月平均透支余额（本币）"],
        ["PD01CJ14"                     , "smul(PD01CJ14, acc_exchange_rate)"   , None  , "R2 最大使用额度（本币）"],
        ["PD01CJ15"                     , "smul(PD01CJ15, acc_exchange_rate)"   , None  , "R3 最大透支余额（本币）"],
        ["monthly_doi_last_repay"       , "day_itvl(PD01CR03, today)"           , None  , "月度还款距今日"],                # Mixed: mixed_doi_last_repay
        ["monthly_doi_report"           , "day_itvl(PD01CR01, today)"           , None  , "月度报告日期距今日"],            # Mixed: mixed_doi_report
        ["last_6m_avg_usd"              , "cb_fst(PD01CJ12, PD01CJ13)"          , None  , "R23账户最近6月平均透支额"],
        ["last_6m_max_usd"              , "cb_fst(PD01CJ14, PD01CJ15)"          , None  , "R23账户最近6月最大透支额"],
        # 最新表现、月度表现根据说明文档混合
        ["mixed_acc_moi_folw"           , "mon_itvl(cb_fst(PD01BR01, PD01AR02), today)"         , None                  , "账户关闭距今月"],
        ["mixed_acc_status"             , "cb_fst(cur_acc_status, monthly_acc_status)"          , None                  , "账户状态"],
        ["mixed_lvl5_status"            , "cb_max(cur_lvl5_status, monthly_lvl5_status)"        , None                  , "账户5级分类"],
        ["mixed_ots"                    , "cb_min(PD01BJ01, PD01CJ01)"                          , None                  , "账户余额"],
        ["mixed_last_repay_amt"         , "cb_fst(PD01BJ02, PD01CJ05)"                          , None                  , "最近实还款"],
        ["mixed_doi_last_repay"         , "cb_min(cur_doi_last_repay, monthly_doi_last_repay)"  , None                  , "最近还款距今日"],
        ["mixed_doi_report"             , "cb_min(cur_doi_report, monthly_doi_report)"          , None                  , "报告时间"],
        # 按月应还 - 包含已结清
        ["alle_mon"                     , "PD01AS01"                                            , "acc_repay_freq == 3" , "全部还款期数（月）"],
        ["alle_mon"                     , "cb_max(mon_itvl(cb_fst(PD01AR02, PD01BR01), PD01AR01), 1)"   , "acc_repay_freq != 3" , "全部还款期数（月）"],
        ["mixed_alle_monthly_repay"     , "sdiv(PD01AJ01, alle_mon)"                            , "acc_cat <= 2"        , "D1R4全周期按月应还款"],
        # 按月应还
        ["folw_mon"                     , "PD01CS01"                                            , "acc_repay_freq == 3" , "剩余还款期数（月）"],
        ["folw_mon"                     , "cb_max(mon_itvl(PD01AR02, PD01CR01), 1)"             , "acc_repay_freq != 3" , "剩余还款期数（月）"],
        # D1R41 月负债：按月还款账户直接取 `PD01CJ04-本月应还款`，否则直接按月直接除
        ["mixed_folw_monthly_repay_"    , "cb_max(PD01CJ04, sdiv(PD01CJ01, folw_mon))"          , "acc_cat <= 3"        , "D1R41按月应还款"],
        ["mixed_folw_monthly_repay"     , "cb_fst(mixed_folw_monthly_repay_, mixed_alle_monthly_repay)" , "acc_cat <= 3", "D1R41按月应还款"],
        # R2 信用卡月负债情况较为复杂：
        # 1. `82-大额专项分期卡` 类似R4账户，可直接用 `PD01CJ04-本月应还款` 计算
        # 2. `81-贷记卡` 可按比例（最低还款比例，5% - 10%）调整 `PD01CJ12-近6个月平均使用额度`
        # 3. 但银行数据报送可能不规范：
        # 3.1. 银行可能直接报 `81-贷记卡`、`PD01AJ02 = 0` 而不是直接报 `82`
        # 3.2. 按报送说明：`PD01CJ12` 应该为近6个月 `PD01CJ02` 均值（对应报送时 `已使用额度`），
        #      指“信用卡循环额度下已使用部分”，但实际报送中大额分期当期应还部分是否占用信用卡
        #      循环额度取决于银行自身，且无论是否包含按比例调整都有问题
        # 3.2.1 若已用额度包含大额专项分期当期应还部分，则调整比例偏低
        # 3.2.2 若已用额度不包含大额专项分期当期应还部分，信用卡大额分期部分负债未被计入
        # 综上，取当月应还与调整后使用额度孰大较好

        # 另别注，人行收入推断也受上述影响：
        # 1. 房贷月供 * [2, 2.5]：一线城市偏 2
        # 2. 信用卡额度额度（平均，其中应剔除额度为 0 账户）按账龄（贷记卡最长账龄），常用
        # 2.1. <12M: 1/4
        # 2.2. 12M - 24M：1/3
        # 2.3. >24M：1/2
        # 在另注，银行、支付宝流水计算收入：
        # 1. 银行流水：max(月均工资, 2000, (近12个月转账收入 - 疑似刷单金额) * 0.2 / 12)
        # 2. 微信：max(12月二维码收入*0.2/12, 12个月支出流水*0.05/12, 2000)
        ["mixed_folw_monthly_repay"     , "cb_max(PD01CJ04, smul(PD01CJ12, 0.1))"                   , "acc_cat == 4"        , "R2按月应还款"],
    ],
    "pboc_credit_info": [
        ["credit_org_cat"               , "map(PD02AD01, org_cat)"              , None  , "授信账户管理机构类型"],
        ["credit_cat"                   , "map(PD02AD02, credit_lmt_cat)"       , None  , "授信额度类型"],
        ["credit_exchange_rate"         , "map(PD02AD03, exchange_rate)"        , None  , "授信协议币种"],
        ["credit_status"                , "map(PD02AD04, credit_protocal_status)"   , None  , "授信协议状态"],
        ["credit_lmt"                   , "smul(PD02AJ01, credit_exchange_rate)"    , None  , "授信额度"],
        ["credit_tmp_lmt"               , "smul(PD02AJ03, credit_exchange_rate)"    , None  , "授信限额"],
        ["credit_ots"                   , "smul(PD02AJ04, credit_exchange_rate)"    , None  , "已用额度"],
        ["credit_moi_start"             , "mon_itvl(PD02AR01, today)"           , None  , "授信起始距今月"],
        ["credit_moi_end"               , "mon_itvl(PD02AR02, today)"           , None  , "授信截至距今月"],
    ],
    "pboc_rel_info": [
        ["rel_org_cat"                  , "map(PD03AD01, org_cat)"              , None  , "相关还款责任管理机构类型"],
        ["rel_biz_cat"                  , "map(PD03AD02, biz_cat)"              , None  , "相关还款责任业务类型"],
        ["rel_lvl5_status"              , "map(PD03AD05, lvl5_status)"          , None  , "相关还款责任5级分类"],
        ["rel_repay_status"             , "map(PD03AD07, repay_status)"         , None  , "相关还款责任还款状态"],
        ["rel_moi_start"                , "mon_itvl(PD03AR01, today)"           , None  , "相关还款责任起始距今"],
        ["rel_moi_end"                  , "mon_itvl(PD03AR02, today)"           , None  , "相关还款责任截至距今"],
    ],
    "pboc_postfee_info": [
        ["postfee_acc_cat"              , "map(PE01AD01, postfee_acc_cat_M)"    , None  , "后付费账户类型"],
        ["postfee_acc_status"           , "map(PE01AD03, postfee_acc_status_M)" , None  , "后付费账户状态"],
    ],
    "pboc_inq_rec": [
        ["inq_rec_org_cat"              , "map(PH010D01, org_cat)"              , None  , "查询机构类型"],
        ["inq_rec_reason_cat"           , "map(PH010Q03, inq_reason_cat)"       , None  , "查询原因类型"],
        ["inq_rec_moi"                  , "mon_itvl(PH010R01, today)"           , None  , "查询距今月"],
    ],
    "pboc_basic_info": [
        ["pi_edu_record"                , "map(PB01AD02, edu_record)"           , None  , "学历"],
        ["pi_edu_degree"                , "map(PB01AD03, edu_degree)"           , None  , "学位"],
        ["pi_emp_job"                   , "map(PB01AD04, emp_status)"           , None  , "就业状况"],
    ],
    "pboc_address": [
        ["pi_res_status"                , "map(PB030D01, res_status)"           , None  , "居住状况"],
    ],
    "pboc_company": [
        ["pi_comp_job"                  , "map(PB040D01, emp_status)"           , None  , "就业状况"],
        ["pi_comp_char"                 , "map(PB040D02, comp_character)"       , None  , "单位性质"],
        ["pi_comp_indust"               , "map(PB040D03, comp_industry)"        , None  , "行业"],
        ["pi_comp_prof"                 , "map(PB040D04, comp_profession)"      , None  , "职业"],
        ["pi_comp_pos"                  , "map(PB040D05, comp_position)"        , None  , "职务"],
        ["pi_comp_prof_title"           , "map(PB040D06, comp_prof_title)"      , None  , "职称"],
    ],
    # 人行征信不在更新公积金信息
    "pboc_housing_fund":[
        ["hf_status"                    , "map(PF05AD01, housing_fund_status)"  , None  , "缴交状态"],
    ],
}


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
    "pboc_postfee_abst":[
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
    "pboc_inq_abst":[
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


# %%
LV20_AGG_CONF = {
    "acc_repay_60m_agg":{
        "part": "acc_repay_60m_agg",
        "lower_from": "acc_repay_60m",
        "from_": "acc_repay_60m,pboc_acc_info",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "{cond}{agg}",
        "agg": {
            "sum": ("{}_sum", "sum({})", "{}之和"),
            "max": ("{}_max", "max({})", "{}最大值"),
            "sum_amt": ("{}_sum", "sum(smul({}, acc_exchange_rate))", "{}之和（本币）"),
            "max_amt": ("{}_max", "max(smul({}, acc_exchange_rate))", "{}最大值（本币）"),
        },
        "cond": {
            "acc_cat": {
                "r2": ("r2"         , "acc_cat == 4"                        , "r2"),
                "d1r41": ("d1r41"   , "acc_cat <= 3"                        , "d1r41"),
                "r23": ("r23"       , "(acc_cat >= 4) & (acc_cat <= 5)"     , "r23"),
            }
        },
        "cros": {
            "sum": ["acc_cat", ],
            "max": ["acc_cat", ],
            "sum_amt": ["acc_cat", ],
            "max_amt": ["acc_cat", ],
        },
    },
    "acc_special_trans_agg":{
        "part": "acc_special_trans_agg",
        "lower_from": "acc_special_trans",
        "from_": "acc_special_trans,pboc_acc_info",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "{cond}{agg}",
        "agg": {
            "sum": ("{}_sum", "sum({})", "{}之和"),
            "max": ("{}_max", "max({})", "{}最大值"),
            "sum_amt": ("{}_sum", "sum(smul({}, acc_exchange_rate))", "{}之和（本币）"),
            "max_amt": ("{}_max", "max(smul({}, acc_exchange_rate))", "{}最大值（本币）"),
        },
        "cond": {
            "acc_cat": {
                "c1": ("c1"         , "acc_cat == 99"                        , "c1"),
                "r2": ("r2"         , "acc_cat == 4"                         , "r2"),
                "d1r41": ("d1r41"   , "acc_cat <= 3"                         , "d1r41"),
                "r23": ("r23"       , "(acc_cat >= 4) & (acc_cat <= 5)"      , "r23"),
            }
        },
        "cros": {
            "sum": ["acc_cat", ],
            "max": ["acc_cat", ],
            "sum_amt": ["acc_cat", ],
            "max_amt": ["acc_cat", ],
        },
    },
    "acc_special_accd_agg": {
        "part": "acc_special_accd_agg",
        "lower_from": "acc_special_accd",
        "from_": "acc_special_accd,pboc_acc_info",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "{cond}{agg}",
        "agg": {
            "sum": ("{}_sum", "sum({})", "之和"),
            "max": ("{}_max", "max({})", "最大值"),
            "sum_amt": ("{}_sum", "sum(smul({}, acc_exchange_rate))", "{}之和（本币）"),
            "max_amt": ("{}_max", "max(smul({}, acc_exchange_rate))", "{}最大值（本币）"),
        },
        "cond": {
            "acc_cat": {
                "r2": ("r2"         , "acc_cat == 4"                        , "r2"),
            }
        },
        "cros": {
            "sum": ["acc_cat", ],
            "max": ["acc_cat", ],
            "sum_amt": ["acc_cat", ],
            "max_amt": ["acc_cat", ],
        },
    },
    "acc_special_insts_agg":{
        "part": "acc_special_insts_agg",
        "lower_from": "acc_special_insts",
        "from_": "acc_special_insts,pboc_acc_info",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "{cond}{agg}",
        "agg": {
            "sum": ("{}_sum", "sum({})", "{}之和"),
            "max": ("{}_max", "max({})", "{}最大值"),
            "sum_amt": ("{}_sum", "sum(smul({}, acc_exchange_rate))", "{}之和（本币）"),
            "max_amt": ("{}_max", "sum(smul({}, acc_exchange_rate))", "{}最大值（本币）"),
        },
        "cond": {
            "acc_cat": {
                "r2": ("r2"             , "acc_cat == 4"                                , "r2"),
                "r281": ("r281"         , "(acc_cat == 4) & (acc_biz_cat == 221)"       , "r281"),
                "r282": ("r282"         , "(acc_cat == 4) & (acc_biz_cat == 231)"       , "r282"),
                # 有些机构会直接报普通贷记卡 `81`、授信额度为 0 做大额专项分期，而不是直接报 `82`
                "r2spec": ("r2spec"     , "((acc_cat == 4) & (acc_biz_cat == 231)) | ((acc_biz_cat == 221) & (PD01AJ02 == 0))"        , "r2spec"),
            }
        },
        "cros": {
            "sum": ["acc_cat", ],
            "max": ["acc_cat", ],
            "sum_amt": ["acc_cat", ],
            "max_amt": ["acc_cat", ],
        },
    },
}


LV2_AGG_CONF = {
    "acc_repay_60m": {
        "part": "acc_repay_60m",
        "from_": "pboc_acc_repay_60_monthly",
        "level": 1,
        "prikey": "rid,certno,accid",
        "key_fmt": "acc_repay_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"(acc_repay_moi >= -{moi}) & (acc_repay_moi <= 0)", f"近{moi}月")
                      for moi in [3, 6, 9, 12, 24, 36, 48]]
                     + [("all", None, "历史"), ]),
            "status": ([(f"eq{rs}", f"acc_repay_status == {rs}", f"还款状态为{rs}")
                       for rs in [1, 2, 3, 4, 5, 6, 7]]
                       + [("le1", "acc_repay_status >= 1", "还款状态大于等于1")]),
            "status_spec": [
                ("gpaid"        , "acc_repay_status_spec == 31"             , "担保人代还"),
                ("asset"        , "acc_repay_status_spec == 32"             , "以资抵债"),
                ("dum"          , "acc_repay_status_spec == 81"             , "呆账"),
            ],
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "期数", ["max", "sum"]),
            "status_max": ("status_max", "max(acc_repay_status)", "最大还款状态", ["max",]),
            "status_ovd_conl_max": ("status_ovd_conl_max", "flat1_max(sortby(acc_repay_status, PD01ER03, 1) > 0)", "最长连续逾期期数", ["max"]),
            "status_sum": ("status_sum", "sum(acc_repay_status)", "还款状态之和", ["max", "sum"]),
            "ovd_max": ("ovd_max", "max(PD01EJ01)", "最大逾期（透支）金额", ["max_amt"]),
            "ovd_sum": ("ovd_sum", "sum(PD01EJ01)", "逾期（透支）金额之和", ["sum_amt", "max_amt"]),
        },
        "cros": [
            (["cnt",]                           , ["mois", "status"]),
            (["cnt",]                           , ["mois", "status_spec"]),
            (["status_max", "status_sum", "status_ovd_conl_max"],
             ["mois", ]),
            (["ovd_max", "ovd_sum"]             , ["mois", "status"]),
            (["ovd_max", "ovd_sum"]             , ["mois", ]),
        ],
    },
    "acc_special_trans": {
        "part": "acc_special_trans",
        "from_": "pboc_acc_special_trans",
        "level": 1,
        "prikey": "rid,certno,accid",
        "key_fmt": "acc_special_trans_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"(acc_special_trans_moi >= -{moi}) & (acc_special_trans_moi <= 0)", f"近{moi}月")
                      for moi in [12, 24, 36, 48]]
                     + [("all", None, "历史"), ]),
            "trans": [("early", "acc_special_trans_type < 10", "提前还款"),
                      ("exhib", "(acc_special_trans_type > 10) & (acc_special_trans_type < 20)", "展期"),
                      ("asset", "(acc_special_trans_type > 20) & (acc_special_trans_type < 30)", "资产剥离、转让、债务减免、平仓"),
                      ("ovd", "acc_special_trans_type > 30", "代偿、以资抵债")]
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "特殊交易记录数", ["max", "sum"]),
            "sum": ("sum", "sum(PD01FJ01)", "特殊交易额度和", ["max_amt", "sum_amt"]),
            "max": ("max", "max(PD01FJ01)", "特殊交易额度最大值", ["max_amt"]),
        },
        "cros": [
            (["cnt", "sum", "max"]              , ["mois", "trans"]),
        ],
    },
    "acc_special_accd": {
        "part": "acc_special_accd",
        "from_": "pboc_acc_special_accd",
        "level": 1,
        "prikey": "rid,certno,accid",
        "key_fmt": "acc_special_accd_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"(acc_special_accd_moi >= -{moi}) & (acc_special_accd_moi <= 0)",
                       f"近{moi}月")
                      for moi in [12, 24, 36, 48]]
                     + [("all", None, "历史"), ]),
            "trans": [("nor", "acc_speical_trans_type <= 1", "正常类"),
                      ("ovd", "acc_special_trans_type >= 2", "逾期类")]
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "特殊事件记录数", ["max", "sum"]),
        },
        "cros": [
            (["cnt", ]                          , ["mois", ]),
        ],
    },
    "acc_special_insts": {
        "part": "acc_special_insts",
        "from_": "pboc_acc_special_insts",
        "level": 1,
        "prikey": "rid,certno,accid",
        "key_fmt": "acc_special_insts_{cond}_{agg}",
        "cond": {
            "mois_start": ([(f"last_{moi}m",
                             f"(acc_special_insts_moi_start >= -{moi}) & (acc_special_insts_moi_start <= 0)",
                             f"近{moi}月开始") for moi in [1, 2, 3, 6, 12, 24, 36, 48]]
                           + [("his", None, "历史"), ]),
            "mois_end": ([(f"folw_{moi}m",
                           f"acc_special_insts_moi_end <= {moi} & acc_special_insts_moi_end > 0",
                           f"未来{moi}月存续") for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "acc_special_insts_moi_end <= 0", "已结清"),
                            ("open", "acc_special_insts_moi_end > 0", "存续")]),
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "大额专项分期记录数", ["max", "sum"]),
            "lmt_sum": ("lmt_sum", "sum(PD01HJ01)", "大额专项分期额度和", ["max_amt", "sum_amt"]),
            "lmt_max": ("lmt_max", "sum(PD01HJ01)", "大额专项分期额度最大值", ["max_amt"]),
            "usd_sum": ("usd_sum", "sum(PD01HJ02)", "大额专项分期已用额度和", ["max_amt", "sum_amt"]),
            "usd_max": ("usd_max", "max(PD01HJ02)", "大额专项分期已用额度最大值", ["max_amt"]),
            "usd_sum_ppor": ("usd_sum_ppor", "sdiv(sum(PD01HJ02), sum(PD01HJ01))", "大额专项分期已用额度占比", ["max"]),
            "monthly_repay_max": ("monthly_repay_max", "max(acc_special_insts_monthly_repay)", "大额分期月还款最大值", ["max_amt"]),
            "monthly_repay_sum": ("monthly_repay_sum", "sum(acc_special_insts_monthly_repay)", "大额分期月还款之和", ["max_amt", "sum_amt"]),
        },
        "cros": [
            (["cnt", "lmt_sum", "lmt_max", "usd_sum", "usd_max", "usd_sum_ppor",
              "monthly_repay_max", "monthly_repay_sum"],
             ["mois_start", ]),
            (["cnt", "lmt_sum", "lmt_max", "usd_sum", "usd_max", "usd_sum_ppor",
              "monthly_repay_max", "monthly_repay_sum"],
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
        "prikey": "rid,certno",
        "key_fmt": "pinfo_mobile_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PB01BR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("all", None, "历史"), ])
        },
        "agg": {
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
        "prikey": "rid,certno",
        "key_fmt": "pinfo_res_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PB030R01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("all", None, "历史"), ]),
            "res_status": [
                ("owned", "pi_res_status < 20", "自有、按揭、共有"),
                ("owned_all", "(pi_res_status > 10) & (pi_res_status <= 13)", "自置、按揭"),
                ("rented", "(pi_res_status > 20) & (pi_res_status < 40)", "租住、借住"),
                ("other", "pi_res_status > 90", "其他"),
            ],
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "居住地数量"),
        },
        "cros": [
            (["cnt", ]                          , ["mois", ]),
            (["cnt", ]                          , ["res_status", ]),

        ]
    },
    "pinfo_comp": {
        "part": "pinfo_comp",
        "from_": "pboc_company",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "pinfo_comp_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PB040R01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("all", None, "历史"), ]),
            "comp_char": [
                ("char_gov", "pi_comp_char == 10", "机关、事业单位"),
                ("char_gov_cap", "pi_comp_char == 20", "国有企业"),
                ("char_fogn_cap", "pi_comp_char == 30", "外资企业"),
                ("char_priv_cap", "pi_comp_char == 40", "个体、私营企业"),
                ("char_other_cap", "pi_comp_char == 50", "其他（三资、民营、团体）"),
                ("char_other", "pi_comp_char == 99", "未知"),
            ],
            "comp_indust": [
                ("1indust", "pi_comp_indust < 20", "第一产业"),
                ("2indust", "(pi_comp_indust > 20) & (pi_comp_indust < 30)", "第二产业"),
                ("3indust", "(pi_comp_indust > 30) & (pi_comp_indust < 90)", "第三产业"),
                ("other_ind", "pi_comp_indust == 99", "其他"),
            ],
            "comp_prof": [
                ("prof_head", "pi_comp_prof == 11", "国家机关、党群组织、企业、事业单位负责人"),
                ("prof_soldier", "pi_comp_prof == 12", "军人"),
                ("prof_prod", "pi_comp_prof == 21", "生产人员"),
                ("prof_prof", "(pi_comp_prof > 30) & (pi_comp_prof < 40)", "技术人员"),
                ("prof_serv", "(pi_comp_prof > 40) & (pi_comp_prof < 50)", "服务人员"),
                ("prof_other", "pi_comp_prof > 90", "其他职业"),
            ],
            "comp_pos": [
                ("pos_sup", "pi_comp_pos == 1", "高级领导"),
                ("pos_mid", "pi_comp_pos == 2", "中级领导"),
                ("pos_inf", "pi_comp_pos == 3", "一般员工"),
                ("pos_other", "pi_comp_pos > 90", "其他职务"),
            ],
            "comp_prof_title": [
                ("pos_title_sup", "pi_comp_prof_title == 1", "高级职称"),
                ("pos_title_mid", "pi_comp_prof_title == 2", "中级职称"),
                ("pos_title_low", "pi_comp_prof_title == 3", "初级职称"),
                ("pos_title_none", "pi_comp_prof_title > 90", "无职称"),
            ],
            "comp_job": [
                ("pi_comp_job", "pi_comp_job == 91", "在职"),
                ("pi_comp_job", "pi_comp_job != 91", "非在职"),
            ],
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "工作单位数量"),
        },
        "cros": [
            (["cnt", ]                          , ["mois", "comp_job"]),
            (["cnt", ]                          , ["comp_char", "comp_job"]),
            (["cnt", ]                          , ["comp_indust", "comp_job"]),
            (["cnt", ]                          , ["comp_prof", "comp_job"]),
            (["cnt", ]                          , ["comp_pos", "comp_job"]),
            (["cnt", ]                          , ["comp_prof_title", "comp_job"]),
        ]
    },
    "acc_no_cat_info": {
        "part": "acc_no_cat_info",
        "from_": "pboc_acc_info",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "acc_{cond}_{agg}",
        "cond": {
            "orgs": [
                ("org_bank", "acc_org_cat == 11", "商业银行"),
                ("org_bank", "acc_org_cat < 20", "银行账户"),
                ("org_nbank", "(acc_org_cat > 20) & (acc_org_cat < 60)", "非银机构账户"),
                ("org_other", "acc_org_cat > 90", "其他机构账户")
            ],
            "biz_cat": [
                ("biz_cat_loan"         , "acc_biz_cat < 200"                           , "贷款业务"),
                ("biz_cat_card"         , "(acc_biz_cat > 200) & (acc_biz_cat < 300)"   , "信用卡业务"),
                ("biz_cat_housing"      , "acc_biz_cat < 120"                           , "房贷业务"),
                ("biz_cat_auto"         , "(acc_biz_cat > 120) & (acc_biz_cat < 130)"   , "车贷业务"),
                ("biz_cat_edu"          , "(acc_biz_cat > 130) & (acc_biz_cat < 140)"   , "教育贷款业务"),
                ("biz_cat_biz"          , "(acc_biz_cat > 140) & (acc_biz_cat < 150)"   , "经营贷、农业贷业务"),
                ("biz_cat_comsu"        , "(acc_biz_cat > 150) & (acc_biz_cat < 160)"   , "消费贷业务"),
                ("biz_cat_security"     , "(acc_biz_cat > 300) & (acc_biz_cat < 400)"   , "证券融资业务"),
                ("biz_cat_leasing"      , "(acc_biz_cat > 400) & (acc_biz_cat < 500)"   , "融资租赁业务"),
                ("biz_cat_dum"          , "acc_biz_cat > 500"                           , "资产处置、垫款业务"),
            ],
            "mois_start": ([(f"last_{moi}m", f"acc_moi_start >= -{moi}", f"近{moi}月开立")
                            for moi in [1, 2, 3, 6, 12, 24, 36, 48]]
                           + [("his", None, "历史"), ]),
            "mois_end": ([(f"folw_{moi}m", f"(acc_moi_end <= {moi}) & (acc_moi_end > 0)", f"预期未来{moi}月存续")
                          for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "acc_moi_end <= 0", "预期关闭"),
                            ("open", "acc_moi_end > 0", "预期存续")]),
            "mixed_acc_status": [
                ("inact"        , "mixed_acc_status < 10"                               , "关闭、未激活、转出"),
                ("nor"          , "mixed_acc_status == 11"                              , "正常"),
                ("ovd"          , "(mixed_acc_status > 20) & (mixed_acc_status < 30)"   , "逾期"),
                ("abnor"        , "(mixed_acc_status > 30) & (mixed_acc_status < 40)"   , "异常"),
                ("dum"          , "mixed_acc_status == 99"                              , "呆账"),
            ],
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "账户数"),
            "ltm_sum": ("lmt_sum", "sum(acc_lmt)", "借款、授信额度之和"),
            "ltm_max": ("lmt_max", "max(acc_lmt)", "借款、授信额度最大值"),
            "mixed_ots_sum": ("ots_sum", "sum(mixed_ots)", "剩余额度之和"),
            "mixed_ots_max": ("ots_max", "max(mixed_ots)", "剩余额度最大值"),
            "mixed_folw_monthly_repay_sum": ("mixed_folw_monthly_repay_sum", "sum(mixed_folw_monthly_repay)", "按月应还款之和"),
            "mixed_folw_monthly_repay_max": ("mixed_folw_monthly_repay_max", "max(mixed_folw_monthly_repay)", "按月应还款最大值"),
        },
        "cros": [
            (["cnt", "ltm_sum", "ltm_max", "mixed_ots_sum", "mixed_ots_max", "mixed_folw_monthly_repay_sum", "mixed_folw_monthly_repay_max"],
             ["orgs", "mois_start"]),
            (["cnt", "ltm_sum", "ltm_max", "mixed_ots_sum", "mixed_ots_max", "mixed_folw_monthly_repay_sum", "mixed_folw_monthly_repay_max"],
             ["orgs", "mois_end"]),
            (["cnt", "ltm_sum", "ltm_max", "mixed_ots_sum", "mixed_ots_max", "mixed_folw_monthly_repay_sum", "mixed_folw_monthly_repay_max"],
             ["orgs", "mixed_acc_status"]),
            (["cnt", "ltm_sum", "ltm_max", "mixed_ots_sum", "mixed_ots_max", "mixed_folw_monthly_repay_sum", "mixed_folw_monthly_repay_max"],
             ["biz_cat", "mois_start"]),
            (["cnt", "ltm_sum", "ltm_max", "mixed_ots_sum", "mixed_ots_max", "mixed_folw_monthly_repay_sum", "mixed_folw_monthly_repay_max"],
             ["biz_cat", "mois_end"]),
            (["cnt", "ltm_sum", "ltm_max", "mixed_ots_sum", "mixed_ots_max", "mixed_folw_monthly_repay_sum", "mixed_folw_monthly_repay_max"],
             ["biz_cat", "mixed_acc_status"]),
        ],
    },
    "acc_cat_info": {
        "part": "acc_cat_info",
        "from_": "pboc_acc_info",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "acc_{cond}_{agg}",
        "cond": {
            "acc_cat": {
                "c1": ("c1"             , "acc_cat == 99"                               , "c1"),
                "d1": ("d1"             , "acc_cat == 1"                                , "d1"),
                "r4": ("r4"             , "acc_cat == 2"                                , "r4"),
                "r1": ("r1"             , "acc_cat == 3"                                , "r1"),
                "r2": ("r2"             , "acc_cat == 4"                                , "r2"),
                "r2cny": ("r2cny"       , "(acc_cat == 4) & (PD01AD04 == \"CNY\")"      , "r2cny"),
                "r3": ("r3"             , "acc_cat == 5"                                , "r3"),
                "d1r4": ("d1r4"         , "acc_cat <= 2"                                , "d1r4"),
                "d1r41": ("d1r41"       , "acc_cat <= 3"                                , "d1r41"),
                "r23": ("r23"           , "(acc_cat >= 4) & (acc_cat <= 5)"             , "r23"),
                "r281": ("r281"         , "(acc_cat == 4) & (acc_biz_cat == 221)"       , "r281"),
                "r282": ("r282"         , "(acc_cat == 4) & (acc_biz_cat == 231)"       , "r282"),
                # 有些机构会直接报普通贷记卡 `81`、授信额度为 0 做大额专项分期，而不是直接报 `82`
                "r2spec": ("r2spec"     , "((acc_cat == 4) & (acc_biz_cat == 231)) | ((acc_biz_cat == 221) & (PD01AJ02 == 0))"        , "r2spec"),
            },
            "mois_start": ([(f"last_{moi}m", f"acc_moi_start >= -{moi}", f"近{moi}月开立")
                            for moi in [1, 2, 3, 6, 9, 12, 18, 24, 36, 48]]
                           + [(None, None, "历史"), ]),
            "mois_end": ([(f"folw_{moi}m", f"(acc_moi_end <= {moi}) & (acc_moi_end > 0)", f"预期未来{moi}月存续")
                          for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "acc_moi_end <= 0", "预期关闭"),
                            ("open", "acc_moi_end > 0", "预期存续")]),
            "mixed_acc_status": [
                ("inact"        , "mixed_acc_status < 10"                               , "关闭、未激活、转出"),
                ("active"       , "mixed_acc_status > 10"                               , "活跃"),
                ("nocls"        , "mixed_acc_status > 0"                                , "未结清"),
                ("nor"          , "mixed_acc_status == 11"                              , "正常"),
                ("ovd"          , "(mixed_acc_status > 20) & (mixed_acc_status < 30)"   , "逾期"),
                ("abnor"        , "(mixed_acc_status > 30) & (mixed_acc_status < 40)"   , "异常"),
                ("dum"          , "mixed_acc_status == 99"                              , "呆账"),
                (None           , None                                                  , None),
            ],
            "mixed_lvl5_status": [
                ("lvl5_nor"         , "mixed_lvl5_status == 1"          , "五级分类正常"),
                ("lvl5_con"         , "mixed_lvl5_status == 2"          , "五级分类关注"),
                ("lvl5_inf"         , "mixed_lvl5_status >= 3"          , "五级分类次级及以上"),
            ],
            "trans_status": [(f"trans_status_eq{ts}", f"acc_trans_status == {ts}",
                              f"转移时状态为{ts}")
                             for ts in [0, 1, 2, 3, 4, 5, 6, 7]],
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "账户数"),
            "lmt_sum": ("lmt_sum", "sum(acc_lmt)", "借款、授信额度之和"),
            "lmt_max": ("lmt_max", "max(acc_lmt)", "借款、授信额度最大值"),
            "lmt_min": ("lmt_min", "min(acc_lmt)", "借款、授信额度最小值"),
            "last_prd_max": ("last_prd_max", "max(-acc_moi_start)", "首个账户距今"),
            # Mixed
            "last_repay_mcnt": ("last_repay_mcnt", "min(mixed_doi_last_repay)", "最近还款距今（天）"),
            "last_repay_amt": ("last_repay_amt", "getn(mixed_last_repay_amt, argmin(mixed_doi_last_repay))", "最近还款金额"),
            "mixed_folw_monthly_repay_sum": ("mixed_folw_monthly_repay_sum", "sum(mixed_folw_monthly_repay)", "按月应还款之和"),
            "mixed_folw_monthly_repay_max": ("mixed_folw_monthly_repay_max", "max(mixed_folw_monthly_repay)", "按月应还款最大值"),
            "mixed_ots_sum": ("ots_sum", "sum(mixed_ots)", "剩余额度之和"),
            "mixed_ots_max": ("ots_max", "max(mixed_ots)", "剩余额度最大值"),
        },
        "cros": [
            # Filter: mixed_acc_status * mois_start
            # Aggs: count, limit, period, outstanding, repayment behavior
            (["cnt",
              "lmt_sum", "lmt_max", "lmt_min",
              "last_prd_max",
              "mixed_ots_sum", "mixed_ots_max",
              "last_repay_mcnt", "last_repay_amt"],
             [("acc_cat", "c1", "d1r41", "r23", "d1", "r4", "r1", "r2", "r2cny", "r3", "r281", "r282", "r2spec"),
              "mixed_acc_status", "mois_start"]),
            # Repayment
            (["mixed_folw_monthly_repay_sum", "mixed_folw_monthly_repay_max"],
             [("acc_cat", "d1r41", "r23", "d1", "r4", "r1", "r2", "r2cny", "r3", "r281", "r282", "r2spec"),
              "mixed_acc_status", "mois_start"]),
            # C1
            (["cnt", "lmt_sum", "lmt_max", "lmt_min", "last_prd_max"],
             ["acc_cat.[c1]", "trans_status"]),
        ],
    },
    "acc_cat_info_d1r41": {
        "part": "acc_cat_info_d1r41",
        "from_": "pboc_acc_info",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "acc_{cond}_{agg}",
        "cond": {
            "acc_cat": {
                "c1": ("c1"             , "acc_cat == 99"                               , "c1"),
                "d1": ("d1"             , "acc_cat == 1"                                , "d1"),
                "r4": ("r4"             , "acc_cat == 2"                                , "r4"),
                "r1": ("r1"             , "acc_cat == 3"                                , "r1"),
                "r2": ("r2"             , "acc_cat == 4"                                , "r2"),
                "r2cny": ("r2cny"       , "(acc_cat == 4) & (PD01AD04 == \"CNY\")"      , "r2cny"),
                "r3": ("r3"             , "acc_cat == 5"                                , "r3"),
                "d1r4": ("d1r4"         , "acc_cat <= 2"                                , "d1r4"),
                "d1r41": ("d1r41"       , "acc_cat <= 3"                                , "d1r41"),
                "r23": ("r23"           , "(acc_cat >= 4) & (acc_cat <= 5)"             , "r23"),
                "r281": ("r281"         , "(acc_cat == 4) & (acc_biz_cat == 221)"       , "r281"),
                "r282": ("r282"         , "(acc_cat == 4) & (acc_biz_cat == 231)"       , "r282"),
                # 有些机构会直接报普通贷记卡 `81`、授信额度为 0 做大额专项分期，而不是直接报 `82`
                "r2spec": ("r2spec"     , "((acc_cat == 4) & (acc_biz_cat == 231)) | ((acc_biz_cat == 221) & (PD01AJ02 == 0))"        , "r2spec"),
            },
            "mois_start": ([(f"last_{moi}m", f"acc_moi_start >= -{moi}", f"近{moi}月开立")
                            for moi in [1, 2, 3, 6, 9, 12, 18, 24, 36, 48]]
                           + [(None, None, "历史"), ]),
            "mois_end": ([(f"folw_{moi}m", f"(acc_moi_end <= {moi}) & (acc_moi_end > 0)", f"预期未来{moi}月存续")
                          for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "acc_moi_end <= 0", "预期关闭"),
                            ("open", "acc_moi_end > 0", "预期存续")]),
            "mixed_acc_status": [
                ("inact"        , "mixed_acc_status < 10"                               , "关闭、未激活、转出"),
                ("active"       , "mixed_acc_status > 10"                               , "活跃"),
                ("nocls"        , "mixed_acc_status > 0"                                , "未结清"),
                ("nor"          , "mixed_acc_status == 11"                              , "正常"),
                ("ovd"          , "(mixed_acc_status > 20) & (mixed_acc_status < 30)"   , "逾期"),
                ("abnor"        , "(mixed_acc_status > 30) & (mixed_acc_status < 40)"   , "异常"),
                ("dum"          , "mixed_acc_status == 99"                              , "呆账"),
                (None           , None                                                  , None),
            ],
            "mixed_lvl5_status": [
                ("lvl5_nor"         , "mixed_lvl5_status == 1"          , "五级分类正常"),
                ("lvl5_con"         , "mixed_lvl5_status == 2"          , "五级分类关注"),
                ("lvl5_inf"         , "mixed_lvl5_status >= 3"          , "五级分类次级及以上"),
            ],
            "trans_status": [(f"trans_status_eq{ts}", f"acc_trans_status == {ts}",
                              f"转移时状态为{ts}")
                             for ts in [0, 1, 2, 3, 4, 5, 6, 7]],
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "账户数"),
            # D1R41
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
            "last_org_cat": ("last_org_cat", "max(argmaxs(acc_doi_start, acc_org_cat))", "最近账户机构类型"),
            "fst_org_cat": ("fst_org_cat", "max(argmins(acc_doi_start, acc_org_cat))", "最早账户机构类型"),
            # D1R412
            "ovd_sum": ("ovd_sum", "sum(PD01CJ06)", "逾期总额之和"),
            "ovd_max": ("ovd_max", "max(PD01CJ06)", "逾期总额最大值"),
            "monthly_folw_prd_max": ("monthly_folw_prd_max", "max(PD01CS01)", "剩余还款期数最大值"),
            "monthly_ovd_prd_sum": ("monthly_ovd_prd_sum", "sum(PD01CS02)", "当前逾期期数之和"),
            "monthly_ovd_prd_max": ("monthly_ovd_prd_max", "max(PD01CS02)", "当前逾期期数最大值"),
        },
        "cros": [
            # D1R41
            (["cnt", ],
             [("acc_cat", "d1r41", "d1r4", "d1", "r4", "r1"),
              "mixed_lvl5_status", "mois_start"]),
            (["m2_ovd_pri_sum", "m2_ovd_pri_max",
              "m3_ovd_pri_sum", "m3_ovd_pri_max",
              "m46_ovd_pri_sum", "m46_ovd_pri_max",
              "m7p_ovd_pri_sum", "m7p_ovd_pri_max",
              "folw_prd_max", "folw_prd_sum",
              "last_org_cat", "fst_org_cat",
              "alle_prd_max", "alle_prd_sum"],
             [("acc_cat", "d1r41", "d1r4", "d1", "r4", "r1"),
              "mixed_acc_status", "mois_start"]),
            # D1R412
            (["ovd_sum", "ovd_max"],
             [("acc_cat", "d1r41", "r2", "r2cny", "r281", "r282", "r2spec"),
              "mois_start"]),
            (["monthly_folw_prd_max", "monthly_ovd_prd_sum", "monthly_ovd_prd_max"],
             [("acc_cat", "d1r41", "r2", "r2cny", "r281", "r282", "r2spec"),
              "mixed_acc_status", "mois_start"]),
         ],
    },
    "acc_cat_info_r23": {
        "part": "acc_cat_info_r23",
        "from_": "pboc_acc_info",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "acc_{cond}_{agg}",
        "cond": {
            "acc_cat": {
                "c1": ("c1"             , "acc_cat == 99"                               , "c1"),
                "d1": ("d1"             , "acc_cat == 1"                                , "d1"),
                "r4": ("r4"             , "acc_cat == 2"                                , "r4"),
                "r1": ("r1"             , "acc_cat == 3"                                , "r1"),
                "r2": ("r2"             , "acc_cat == 4"                                , "r2"),
                "r2cny": ("r2cny"       , "(acc_cat == 4) & (PD01AD04 == \"CNY\")"      , "r2cny"),
                "r3": ("r3"             , "acc_cat == 5"                                , "r3"),
                "d1r4": ("d1r4"         , "acc_cat <= 2"                                , "d1r4"),
                "d1r41": ("d1r41"       , "acc_cat <= 3"                                , "d1r41"),
                "r23": ("r23"           , "(acc_cat >= 4) & (acc_cat <= 5)"             , "r23"),
                "r281": ("r281"         , "(acc_cat == 4) & (acc_biz_cat == 221)"       , "r281"),
                "r282": ("r282"         , "(acc_cat == 4) & (acc_biz_cat == 231)"       , "r282"),
                # 有些机构会直接报普通贷记卡 `81`、授信额度为 0 做大额专项分期，而不是直接报 `82`
                "r2spec": ("r2spec"     , "((acc_cat == 4) & (acc_biz_cat == 231)) | ((acc_biz_cat == 221) & (PD01AJ02 == 0))"        , "r2spec"),
            },
            "mois_start": ([(f"last_{moi}m", f"acc_moi_start >= -{moi}", f"近{moi}月开立")
                            for moi in [1, 2, 3, 6, 9, 12, 18, 24, 36, 48]]
                           + [(None, None, "历史"), ]),
            "mois_end": ([(f"folw_{moi}m", f"(acc_moi_end <= {moi}) & (acc_moi_end > 0)", f"预期未来{moi}月存续")
                          for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "acc_moi_end <= 0", "预期关闭"),
                            ("open", "acc_moi_end > 0", "预期存续")]),
            "mixed_acc_status": [
                ("inact"        , "mixed_acc_status < 10"                               , "关闭、未激活、转出"),
                ("active"       , "mixed_acc_status > 10"                               , "活跃"),
                ("nocls"        , "mixed_acc_status > 0"                                , "未结清"),
                ("nor"          , "mixed_acc_status == 11"                              , "正常"),
                ("ovd"          , "(mixed_acc_status > 20) & (mixed_acc_status < 30)"   , "逾期"),
                ("abnor"        , "(mixed_acc_status > 30) & (mixed_acc_status < 40)"   , "异常"),
                ("dum"          , "mixed_acc_status == 99"                              , "呆账"),
                (None           , None                                                  , None),
            ],
            "mixed_lvl5_status": [
                ("lvl5_nor"         , "mixed_lvl5_status == 1"          , "五级分类正常"),
                ("lvl5_con"         , "mixed_lvl5_status == 2"          , "五级分类关注"),
                ("lvl5_inf"         , "mixed_lvl5_status >= 3"          , "五级分类次级及以上"),
            ],
            "trans_status": [(f"trans_status_eq{ts}", f"acc_trans_status == {ts}",
                              f"转移时状态为{ts}")
                             for ts in [0, 1, 2, 3, 4, 5, 6, 7]],
        },
        "agg": {
            # R2
            "monthly_usd_sum": ("monthly_usd_sum", "sum(PD01CJ02)", "月度已用额度之和"),
            "monthly_usd_max": ("monthly_usd_max", "max(PD01CJ02)", "月度已用额度最大值"),
            "monthly_special_insts_sum": ("monthly_special_insts_sum", "sum(PD01CJ03)", "月度未出单大额专项余额之和"),
            "monthly_special_insts_max": ("monthly_special_insts_max", "max(PD01CJ03)", "月度未出单大额专项余额最大值"),
            # R3
            "m7p_ovd_sum": ("m7p_ovd_sum", "sum(PD01CJ11)", "m7p透支未付余额之和"),
            "m7p_ovd_max": ("m7p_ovd_max", "sum(PD01CJ11)", "m7p透支未付余额最大值"),
            # R23
            "last_6m_avg_usd_sum": ("last_6m_avg_usd_sum", "sum(last_6m_avg_usd)", "最近6个月平均使用额度之和"),
            "last_6m_avg_usd_max": ("last_6m_avg_usd_max", "sum(last_6m_avg_usd)", "最近6个月平均使用额度最大值"),
            "last_6m_max_usd_max": ("last_6m_max_usd_max", "sum(last_6m_max_usd)", "最近6个月最大使用额度最大值"),
        },
        "cros": [
            # R2
            (["monthly_usd_sum", "monthly_usd_max",
              "monthly_special_insts_sum", "monthly_special_insts_max"],
             [("acc_cat", "r2", "r2cny", "r281", "r282", "r2spec"),
              "mixed_acc_status", "mois_start"]),
            # R3
            (["m7p_ovd_sum", "m7p_ovd_max"],
             [("acc_cat", "r3", ),
              "mixed_acc_status", "mois_start"]),
            # R23
            (["last_6m_avg_usd_sum", "last_6m_avg_usd_max", "last_6m_max_usd_max"],
             [("acc_cat", "r23", "r2", "r2cny", "r281", "r282", "r2spec"),
              "mixed_acc_status", "mois_start"]),
        ],
    },
    "credit_info": {
        "part": "credit_info",
        "from_": "pboc_credit_info",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "credit_{cond}_{agg}",
        "cond": {
            "orgs": [
                ("org_bank", "credit_org_cat < 20", "银行账户"),
                ("org_nbank", "(credit_org_cat > 20) & (credit_org_cat < 60)", "非银机构账户"),
                ("org_other", "credit_org_cat > 90", "其他机构账户")
            ],
            "mois_start": ([(f"last_{moi}m", f"credit_moi_start >= -{moi}", f"近{moi}月开始")
                            for moi in [1, 2, 3, 6, 12, 24, 36, 48]]
                           + [("his", None, "历史"), ]),
            "mois_end": ([(f"folw_{moi}m", f"(credit_moi_end <= {moi}) & (credit_moi_end > 0)", f"未来{moi}月存续")
                          for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "credit_moi_end <= 0", "已结束"),
                            ("open", "credit_moi_end > 0", "存续")]),
            "credit_cat": [
                ("rev", "credit_cat == 10", "循环贷授信额度"),
                ("norev", "credit_cat == 20", "非循环贷授信额度"),
                ("card", "credit_cat >= 30", "信用卡独立、共享授信额度"),
            ],
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "授信数量"),
            "lmt_sum": ("lmt_sum", "sum(smul(PD02AJ01, credit_exchange_rate))", "授信额度之和"),
            "lmt_max": ("lmt_max", "max(smul(PD02AJ01, credit_exchange_rate))", "授信额度最大值"),
            "lmt_min": ("lmt_min", "max(smul(PD02AJ01, credit_exchange_rate))", "授信额度最小值"),
            "lmt2_sum": ("lmt2_sum", "sum(smul(PD02AJ03, credit_exchange_rate))", "授信限额之和"),
            "lmt2_max": ("lmt2_max", "max(smul(PD02AJ03, credit_exchange_rate))", "授信限额最大值"),
            "usd_sum": ("usd_sum", "sum(smul(PD02AJ04, credit_exchange_rate))", "已用额度之和"),
            "usd_max": ("usd_max", "max(smul(PD02AJ04, credit_exchange_rate))", "已用额度最大值"),
        },
        "cros": [
            [["cnt", "lmt_sum", "lmt_max", "lmt_min", "lmt2_sum", "lmt2_max", "usd_sum", "usd_max"],
             ["mois_start", "orgs"]],
            [["cnt", "lmt_sum", "lmt_max", "lmt_min", "lmt2_sum", "lmt2_max", "usd_sum", "usd_max"],
             ["mois_end", "orgs"]],
            [["cnt", "lmt_sum", "lmt_max", "lmt_min", "lmt2_sum", "lmt2_max", "usd_sum", "usd_max"],
             ["mois_start", "credit_cat"]],
            [["cnt", "lmt_sum", "lmt_max", "lmt_min", "lmt2_sum", "lmt2_max", "usd_sum", "usd_max"],
             ["mois_end", "credit_cat"]],
        ]
    },
    "rel_info": {
        "from_": "pboc_rel_info",
        "part": "rel_info",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "rel_{cond}_{agg}",
        "cond": {
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
                            for moi in [1, 2, 3, 6, 12, 18, 24, 36, 48]]
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
        "agg": {
            "cnt": ("cnt", "count(_)", "相关还款责任数"),
            "respon_sum": ("resp_sum", "sum(PD03AJ01)", "相关责任金额之和"),
            "respon_max": ("resp_max", "max(PD03AJ01)", "相关责任金额最大值"),
            "acc_sum": ("acc_sum", "sum(PD03AJ02)", "相关责任账户金额之和"),
            "acc_max": ("acc_max", "max(PD03AJ02)", "相关责任账户金额最大值"),
            "repay_status_max": ("repay_status_max", "max(rel_repay_status)", "相关责任逾期月数最大值"),
            "repay_status_sum": ("repay_status_sum", "sum(rel_repay_status)", "相关责任逾期月数之和"),
            "ovd_month_max": ("ovd_month_max", "max(PD03AS01)", "相关责任逾期月数最大值"),
            "ovd_month_sum": ("ovd_month_sum", "sum(PD03AS01)", "相关责任逾期月数之和"),
            "folw_prd_max": ("folw_prd_max", "max(rel_moi_end)", "相关责任剩余最长期限"),
            "last_prd_max": ("last_prd_max", "max(-rel_moi_start)", "相关责任首个账户距今"),
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
        "prikey": "rid,certno",
        "key_fmt": "postfee_{cond}_{agg}",
        "cond": {
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
        "agg": {
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
        "prikey": "rid,certno",
        "key_fmt": "tax_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF01AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
        },
        "agg": {
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
        "prikey": "rid,certno",
        "key_fmt": "lawsuit_{cond}_{agg}",
        "cond":{
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF02AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "None": [("all", None, "所有记录中"), ]
        },
        "agg": {
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
        "prikey": "rid,certno",
        "key_fmt": "enforcement_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF03AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "None": [("all", None, "所有记录中"), ]
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
            (["prd_max", "prd_min"]                 , ["None"]),
        ],
    },
    "gov_punishment": {
        "part": "gov_punishment",
        "from_": "pboc_gov_punishment",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "gov_punishment_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF04AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [6, 12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "None": [("all", None, "所有记录中"), ]

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
    },
    "housing_fund": {
        "part": "housing_fund",
        "from_": "pboc_housing_fund",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "housing_fund_{cond}_{agg}",
        "cond": {
            "mois_start": ([(f"last_{moi}m", f"mon_itvl(PF05AR02, today) >= -{moi}",
                             f"近{moi}月") for moi in [3, 6, 12, 24, 36, 48]]
                           + [("his", None, "历史"), ]),
            "mois_end": ([(f"last_{moi}m",
                           f"(mon_itvl(PF05AR03, today) <= {moi}) & (mon_itvl(PF05AR03, today) > 0)",
                           f"未来{moi}月存续") for moi in [6, 12, 24, 36, 48]]
                         + [("closed", "mon_itvl(PF05AR03, today) <= 0", "已结束"),
                            ("open", "mon_itvl(PF05AR03, today) > 0", "存续")]),
            "hf_status": [
                ("active", "hf_status == 1", "缴交"),
                ("frozen", "hf_status == 2", "封存"),
                ("closed", "hf_status == 3", "销户"),
            ],
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "住房公积金账户数量"),
            "sum": ("sum", "sum(PF05AJ01)", "住房公积金月缴存额之和"),
            "max": ("max", "max(PF05AJ01)", "住房公积金月缴存额最大值"),
            "start_prd_max": ("start_prd_max", "max(mon_itvl(today, PF05AR02))", "最早初缴距今"),
            "start_prd_min": ("start_prd_min", "min(mon_itvl(today, PF05AR02))", "最晚初缴距今"),
            "end_prd_max": ("end_prd_max", "max(mon_itvl(today, PF05AR03))", "最晚缴至距今"),
            "end_prd_min": ("end_prd_min", "min(mon_itvl(today, PF05AR03))", "最早缴至距今"),
            "latest_prd_min": ("latest_prd_min", "min(mon_itvl(today, PF05AR04))", "最近缴交距今"),
        },
        "cros": [
            (["cnt", "sum", "max"]                  , ["mois_start", "hf_status"]),
            (["cnt", "sum", "max"]                  , ["mois_end", "hf_status"]),
            (["cnt", "sum", "max"]                  , ["mois_end"]),
            (["start_prd_max", "start_prd_min",
              "end_prd_max", "end_prd_min",
              "latest_prd_min"],
             ["hf_status"])
        ],
    },
    "allowance": {
        "part": "sub_allowance",
        "from_": "pboc_sub_allowance",
        "level": 0,
        "prikey": "rid,certno",
        "key_fmt": "allowance_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF06AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [12, 24, 36, 48]]
                     + [("his", None, "历史"), ])
        },
        "agg": {
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
        "prikey": "rid,certno",
        "key_fmt": "cert_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF07AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "None": [("all", None, "所有记录中"), ]
        },
        "agg": {
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
        "prikey": "rid,certno",
        "key_fmt": "award_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PF08AR01, today) >= -{moi}",
                       f"近{moi}月") for moi in [12, 24, 36, 48]]
                     + [("his", None, "历史"), ]),
            "None": [("all", None, "所有记录中"), ]
        },
        "agg": {
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
        "prikey": "rid,certno",
        "key_fmt": "inq_rec_{cond}_{agg}",
        "cond": {
            "mois": ([(f"last_{moi}m", f"mon_itvl(PH010R01, today) >= -{moi}",
                       f"近{moi}月") for moi in [1, 2, 3, 6, 9, 12, 18, 24, 36, 48]]
                     + [("his", None, "历史"), ]
                     + [(f"last_{doi}d", f"day_itvl(PH010R01, today) >= -{doi}",
                         f"近{doi}日") for doi in [0, 1, 2, 3, 5, 10, 30]]),
            "orgs": [
                ("org_bank", "inq_rec_org_cat < 20", "银行账户"),
                ("org_nbank", "(inq_rec_org_cat > 20) & (inq_rec_org_cat < 60)", "非银机构账户"),
                ("org_other", "inq_rec_org_cat > 90", "其他机构账户"),
                ("dorg_combank", "inq_rec_org_cat == 11", "商业银行"),
                ("dorg_conbank", "inq_rec_org_cat == 12", "村镇银行"),
                ("dorg_leasing", "inq_rec_org_cat == 22", "融资租赁"),
                ("dorg_autofin", "inq_rec_org_cat == 23", "汽金公司"),
                ("dorg_comsufin", "inq_rec_org_cat == 24", "消金公司"),
                ("dorg_loan", "inq_rec_org_cat == 25", "贷款公司"),
                ("dorg_security", "inq_rec_org_cat == 31", "证券公司"),
                ("dorg_insur", "inq_rec_org_cat == 41", "保险"),
                ("dorg_sloan", "inq_rec_org_cat == 51", "小额贷款公司"),
                ("dorg_guar", "inq_rec_org_cat == 53", "融担公司"),
            ],
            "inq_reason": [
                ("for_loan", "inq_rec_reason_cat == 11", "贷前审批_贷款"),
                ("for_card", "inq_rec_reason_cat == 12", "贷前审批_信用卡"),
                ("for_guar", "inq_rec_reason_cat == 13", "贷前审批_担保资格审查"),
                ("for_leasing", "inq_rec_reason_cat == 17", "贷前审批_融资审批"),
                ("for_lmt", "inq_rec_reason_cat == 18", "贷前审批_额度审批"),
                ("for_pre", "inq_rec_reason_cat < 20", "贷前审批"),
                ("for_after", "(inq_rec_reason_cat > 20) & (inq_rec_reason_cat < 30)", "贷后管理"),
                ("for_rel", "(inq_rec_reason_cat > 30) & (inq_rec_reason_cat < 40)", "关联审查"),
                ("for_others", "inq_rec_reason_cat > 40", "其他原因审查")
            ],
        },
        "agg": {
            "cnt": ("cnt", "count(_)", "查询数量"),
            "last_6in24m_coef_var": ("coef_var", "coef_var(hist(inq_rec_moi, [-24, -18, -12, -6, 0]))", "近24个月查询量变异系数"),
        },
        "cros": [
            [["cnt",]                               , ["mois", "orgs", "inq_reason"]],
            [["cnt",]                               , ["mois", "orgs",]],
            [["cnt",]                               , ["mois", "inq_reason",]],
            [["last_6in24m_coef_var",]              , ["orgs", "inq_reason"]],
            [["last_6in24m_coef_var",]              , ["orgs",]],
            [["last_6in24m_coef_var",]              , ["inq_reason",]],
        ],
    },
}
