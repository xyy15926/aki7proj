#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: confflat8.py
#   Author: xyy15926
#   Created: 2024-09-14 14:42:54
#   Updated: 2024-10-24 09:16:23
#   Description:
# ---------------------------------------------------------

# %%
PBOC_BASIC_INFO = {
    "part": "pboc_basic_info",
    "desc": "基本信息",
    "steps": [
        {
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
            ]
        }
    ],
    "prikey": ["rid", "certno"],
    "level": 0,
    "fields": [
        ["PA01AI01", "PRH:PA01:PA01A:PA01AI01", "VARCHAR(255)", "报告编号"],
        ["PA01AR01", "PRH:PA01:PA01A:PA01AR01", "DATE", "报告时间"],
        ["PA01BQ01", "PRH:PA01:PA01B:PA01BQ01", "VARCHAR(1022)", "被查询者姓名"],
        ["PA01BD01", "PRH:PA01:PA01B:PA01BD01", "VARCHAR(31)", "被查询者证件类型"],
        ["PA01BI01", "PRH:PA01:PA01B:PA01BI01", "VARCHAR(255)", "被查询者证件号"],
        ["PA01BI02", "PRH:PA01:PA01B:PA01BI02", "VARCHAR(255)", "查询机构代码"],
        ["PA01BD02", "PRH:PA01:PA01B:PA01BD02", "VARCHAR(31)", "查询原因代码"],
        ["PA01CH", "PRH:PA01:PA01C:PA01CH", "TEXT", "其他证件信息"],
        ["PA01DH", "PRH:PA01:PA01D:PA01DH", "TEXT", "防欺诈警示信息"],
        ["PA01ES01", "PRH:PA01:PA01E:PA01ES01", "INT", "异议标注数目"],
        ["PB01AD01", "PIM:PB01:PB01A:PB01AD01", "VARCHAR(31)", "性别"],
        ["PB01AR01", "PIM:PB01:PB01A:PB01AR01", "DATE", "出生日期"],
        ["PB01AD02", "PIM:PB01:PB01A:PB01AD02", "VARCHAR(31)", "学历"],
        ["PB01AD03", "PIM:PB01:PB01A:PB01AD03", "VARCHAR(31)", "学位"],
        ["PB01AD04", "PIM:PB01:PB01A:PB01AD04", "VARCHAR(31)", "就业状况"],
        ["PB01AD05", "PIM:PB01:PB01A:PB01AD05", "VARCHAR(31)", "国籍"],
        ["PB01AQ01", "PIM:PB01:PB01A:PB01AQ01", "VARCHAR(1022)", "电子邮箱"],
        ["PB01AQ02", "PIM:PB01:PB01A:PB01AQ02", "VARCHAR(1022)", "通讯地址"],
        ["PB01AQ03", "PIM:PB01:PB01A:PB01AQ03", "VARCHAR(1022)", "户籍地址"],
        ["PB01BS01", "PIM:PB01:PB01B:PB01BS01", "INT", "手机号记录数量"],
        ["PB020D01", "PMM:PB02:PB020D01", "VARCHAR(31)", "婚姻状态"],
        ["PB020D02", "PMM:PB02:PB020D02", "VARCHAR(31)", "配偶证件类型"],
        ["PB020I01", "PMM:PB02:PB020I01", "VARCHAR(255)", "配偶证件号"],
        ["PB020Q01", "PMM:PB02:PB020Q01", "VARCHAR(1022)", "配偶姓名"],
        ["PB020Q02", "PMM:PB02:PB020Q02", "VARCHAR(1022)", "配偶工作单位"],
        ["PB020Q03", "PMM:PB02:PB020Q03", "VARCHAR(1022)", "配偶联系电话"],
    ],
}


# %%
PBOC_MOBILE = {
    "part": "pboc_mobile",
    "desc": "手机号",
    "steps": [
        {
            "content": "PIM:PB01:PB01B:PB01BH:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PB01BQ01", "PB01BQ01", "VARCHAR(1022)", "手机号"],
        ["PB01BR01", "PB01BR01", "DATE", "信息更新日期"],
    ],
}


# %%
PBOC_ADDRESS = {
    "part": "pboc_address",
    "desc": "居住地址",
    "steps": [
        {
            "content": "PRM:PB03:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PB030Q01", "PB030Q01", "VARCHAR(1022)", "居住地址"],
        ["PB030Q02", "PB030Q02", "VARCHAR(1022)", "住宅电话"],
        ["PB030D01", "PB030D01", "VARCHAR(31)", "居住状况"],
        ["PB030R01", "PB030R01", "DATE", "信息更新日期"],
    ],
}


# %%
PBOC_COMPANY = {
    "part": "pboc_company",
    "desc": "工作单位",
    "steps": [
        {
            "content": "POM:PB04:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PB040D01", "PB040D01", "VARCHAR(31)", "就业状况"],
        ["PB040D02", "PB040D02", "VARCHAR(31)", "单位性质"],
        ["PB040D04", "PB040D04", "VARCHAR(31)", "职业"],
        ["PB040D03", "PB040D03", "VARCHAR(31)", "行业"],
        ["PB040D05", "PB040D05", "VARCHAR(31)", "职务"],
        ["PB040D06", "PB040D06", "VARCHAR(31)", "职称"],
        ["PB040Q01", "PB040Q01", "VARCHAR(1022)", "工作单位"],
        ["PB040Q02", "PB040Q02", "VARCHAR(1022)", "单位地址"],
        ["PB040Q03", "PB040Q03", "VARCHAR(1022)", "单位电话"],
        ["PB040R01", "PB040R01", "DATE", "进入本单位年份"],
        ["PB040R02", "PB040R02", "DATE", "信息更新日期"],
    ],
}


# %%
PBOC_SCORE = {
    "part": "pboc_score",
    "desc": "评分",
    "steps": [
        {
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
            ]
        }
    ],
    "prikey": ["rid", "certno"],
    "level": 0,
    "fields": [
        ["PC010Q01", "PSM:PC01:PC010Q01", "INT", "数字解读"],
        ["PC010Q02", "PSM:PC01:PC010Q02", "INT", "相对位置"],
        ["PC010D01", "PSM:PC01:PC010D01", "VARCHAR(1022)", "分数说明"],
    ],
}


# %%
PBOC_BIZ_ABST = {
    "part": "pboc_biz_abst",
    "desc": "信贷业务概要",
    "steps": [
        {
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
            ]
        }
    ],
    "prikey": ["rid", "certno"],
    "level": 0,
    "fields": [
        [
            "PC02AD01_11",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD01&&PC02AD01=="11"',
            "VARCHAR(31)",
            "业务类型:11 个人住房贷款",
        ],
        [
            "PC02AD02_11",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD02&&PC02AD01=="11"',
            "VARCHAR(31)",
            "业务类型:11 个人住房贷款|业务大类：贷款",
        ],
        [
            "PC02AR02_11",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AR01&&PC02AD01=="11"',
            "DATE",
            "首笔住房贷款发放月份",
        ],
        [
            "PC02AS03_11",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AS03&&PC02AD01=="11"',
            "INT",
            "住房贷款账户数",
        ],
        [
            "PC02AD01_12",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD01&&PC02AD01=="12"',
            "VARCHAR(31)",
            "业务类型:12 个人商用房（包括商住两用房）贷款",
        ],
        [
            "PC02AD02_12",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD02&&PC02AD01=="12"',
            "VARCHAR(21)",
            "业务类型:12 个人商用房（包括商住两用房）贷款|业务大类：贷款",
        ],
        [
            "PC02AR02_12",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AR01&&PC02AD01=="12"',
            "DATE",
            "首笔商用房贷款发放月份",
        ],
        [
            "PC02AS03_12",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AS03&&PC02AD01=="12"',
            "INT",
            "商用房贷款账户数",
        ],
        [
            "PC02AD01_19",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD01&&PC02AD01=="19"',
            "VARCHAR(31)",
            "业务类型:19 其他类贷款",
        ],
        [
            "PC02AD02_19",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD02&&PC02AD01=="19"',
            "VARCHAR(31)",
            "业务类型:19 其他类贷款|业务大类：贷款",
        ],
        [
            "PC02AR02_19",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AR01&&PC02AD01=="19"',
            "DATE",
            "首笔其他类贷款发放月份",
        ],
        [
            "PC02AS03_19",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AS03&&PC02AD01=="19"',
            "INT",
            "其他类贷款账户数",
        ],
        [
            "PC02AD01_21",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD01&&PC02AD01=="21"',
            "VARCHAR(31)",
            "业务类型:21 贷记卡",
        ],
        [
            "PC02AD02_21",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD02&&PC02AD01=="21"',
            "VARCHAR(31)",
            "业务类型:21 贷记卡|业务大类：信用卡",
        ],
        [
            "PC02AR02_21",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AR01&&PC02AD01=="21"',
            "DATE",
            "贷记卡首笔业务发放月份",
        ],
        [
            "PC02AS03_21",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AS03&&PC02AD01=="21"',
            "INT",
            "贷记卡账户数",
        ],
        [
            "PC02AD01_22",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD01&&PC02AD01=="22"',
            "VARCHAR(31)",
            "业务类型:22 准贷记卡",
        ],
        [
            "PC02AD02_22",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD02&&PC02AD01=="22"',
            "VARCHAR(31)",
            "业务类型:22 准贷记卡|业务大类：信用卡",
        ],
        [
            "PC02AR02_22",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AR01&&PC02AD01=="22"',
            "DATE",
            "准贷记卡首笔业务发放月份",
        ],
        [
            "PC02AS03_22",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AS03&&PC02AD01=="22"',
            "INT",
            "准贷记卡账户数",
        ],
        [
            "PC02AD01_99",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD01&&PC02AD01=="99"',
            "VARCHAR(31)",
            "业务类型:99 其他",
        ],
        [
            "PC02AD02_99",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AD02&&PC02AD01=="99"',
            "VARCHAR(31)",
            "业务类型:99 其他|业务大类：其他",
        ],
        [
            "PC02AR02_99",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AR01&&PC02AD01=="99"',
            "DATE",
            "首笔其他信贷账户业务发放月份",
        ],
        [
            "PC02AS03_99",
            'PCO:PC02:PC02A:PC02AH:[getn(nnfilter(_), 0)]:PC02AS03&&PC02AD01=="99"',
            "INT",
            "其他信贷账户业务账户数",
        ],
        ["PC02AS01", "PCO:PC02:PC02A:PC02AS01", "INT", "账户数合计"],
        ["PC02AS02", "PCO:PC02:PC02A:PC02AS02", "INT", "业务类型记录数"],
    ],
}


# %%
PBOC_CACC_ABST = {
    "part": "pboc_cacc_abst",
    "desc": "C类账户概要",
    "steps": [
        {
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
            ]
        }
    ],
    "prikey": ["rid", "certno"],
    "level": 0,
    "fields": [
        [
            "PC02BD01_1",
            'PCO:PC02:PC02B:PC02BH:[getn(nnfilter(_), 0)]:PC02BD01&&PC02BD01=="1"',
            "VARCHAR(31)",
            "业务类型：资产处置业务",
        ],
        [
            "PC02BS03_1",
            'PCO:PC02:PC02B:PC02BH:[getn(nnfilter(_), 0)]:PC02BS03&&PC02BD01=="1"',
            "INT",
            "资产处置业务账户数",
        ],
        [
            "PC02BJ02_1",
            'PCO:PC02:PC02B:PC02BH:[getn(nnfilter(_), 0)]:PC02BJ02&&PC02BD01=="1"',
            "INT",
            "资产处置业务账户余额",
        ],
        [
            "PC02BD01_2",
            'PCO:PC02:PC02B:PC02BH:[getn(nnfilter(_), 0)]:PC02BD01&&PC02BD01=="2"',
            "VARCHAR(31)",
            "业务类型：垫款业务",
        ],
        [
            "PC02BS03_2",
            'PCO:PC02:PC02B:PC02BH:[getn(nnfilter(_), 0)]:PC02BS03&&PC02BD01=="2"',
            "INT",
            "垫款业务账户数",
        ],
        [
            "PC02BJ02_2",
            'PCO:PC02:PC02B:PC02BH:[getn(nnfilter(_), 0)]:PC02BJ02&&PC02BD01=="2"',
            "INT",
            "垫款业务账户余额",
        ],
        ["PC02BJ01", "PCO:PC02:PC02B:PC02BJ01", "INT", "账户余额合计"],
        ["PC02BS01", "PCO:PC02:PC02B:PC02BS01", "INT", "账户数合计"],
        ["PC02BS02", "PCO:PC02:PC02B:PC02BS02", "INT", "C类账户记录数"],
    ],
}


# %%
PBOC_DUM_ABST = {
    "part": "pboc_dum_abst",
    "desc": "呆账概要",
    "steps": [
        {
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
            ]
        }
    ],
    "prikey": ["rid", "certno"],
    "level": 0,
    "fields": [
        ["PC02CJ01", "PCO:PC02:PC02C:PC02CJ01", "INT", "呆账账户余额"],
        ["PC02CS01", "PCO:PC02:PC02C:PC02CS01", "INT", "呆账账户数"],
    ],
}


# %%
PBOC_DRACC_ABST = {
    "part": "pboc_dracc_abst",
    "desc": "DR类账户概要",
    "steps": [
        {
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
            ]
        }
    ],
    "prikey": ["rid", "certno"],
    "level": 0,
    "fields": [
        [
            "PC02DD01_1",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DD01&&PC02DD01=="1"',
            "VARCHAR(31)",
            "账户类型D1",
        ],
        [
            "PC02DS02_1",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS02&&PC02DD01=="1"',
            "INT",
            "逾期D1账户数",
        ],
        [
            "PC02DS03_1",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS03&&PC02DD01=="1"',
            "INT",
            "逾期D1月份数",
        ],
        [
            "PC02DJ01_1",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DJ01&&PC02DD01=="1"',
            "INT",
            "逾期D1单月最高逾期（透支）总额",
        ],
        [
            "PC02DS04_1",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS04&&PC02DD01=="1"',
            "INT",
            "逾期D1最长逾期（透支）月数",
        ],
        [
            "PC02DD01_2",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DD01&&PC02DD01=="2"',
            "VARCHAR(31)",
            "账户类型R4",
        ],
        [
            "PC02DS02_2",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS02&&PC02DD01=="2"',
            "INT",
            "逾期R4账户数",
        ],
        [
            "PC02DS03_2",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS03&&PC02DD01=="2"',
            "INT",
            "逾期R4月份数",
        ],
        [
            "PC02DJ01_2",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DJ01&&PC02DD01=="2"',
            "INT",
            "逾期R4单月最高逾期（透支）总额",
        ],
        [
            "PC02DS04_2",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS04&&PC02DD01=="2"',
            "INT",
            "逾期R4最长逾期（透支）月数",
        ],
        [
            "PC02DD01_3",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DD01&&PC02DD01=="3"',
            "VARCHAR(31)",
            "账户类型R1",
        ],
        [
            "PC02DS02_3",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS02&&PC02DD01=="3"',
            "INT",
            "逾期R1账户数",
        ],
        [
            "PC02DS03_3",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS03&&PC02DD01=="3"',
            "INT",
            "逾期R1月份数",
        ],
        [
            "PC02DJ01_3",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DJ01&&PC02DD01=="3"',
            "INT",
            "逾期R1单月最高逾期（透支）总额",
        ],
        [
            "PC02DS04_3",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS04&&PC02DD01=="3"',
            "INT",
            "逾期R1最长逾期（透支）月数",
        ],
        [
            "PC02DD01_4",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DD01&&PC02DD01=="4"',
            "VARCHAR(31)",
            "账户类型R2",
        ],
        [
            "PC02DS02_4",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS02&&PC02DD01=="4"',
            "INT",
            "逾期R2账户数",
        ],
        [
            "PC02DS03_4",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS03&&PC02DD01=="4"',
            "INT",
            "逾期R2月份数",
        ],
        [
            "PC02DJ01_4",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DJ01&&PC02DD01=="4"',
            "INT",
            "逾期R2单月最高逾期（透支）总额",
        ],
        [
            "PC02DS04_4",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS04&&PC02DD01=="4"',
            "INT",
            "逾期R2最长逾期（透支）月数",
        ],
        [
            "PC02DD01_5",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DD01&&PC02DD01=="5"',
            "VARCHAR(31)",
            "账户类型R3",
        ],
        [
            "PC02DS02_5",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS02&&PC02DD01=="5"',
            "INT",
            "逾期R3账户数",
        ],
        [
            "PC02DS03_5",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS03&&PC02DD01=="5"',
            "INT",
            "逾期R3月份数",
        ],
        [
            "PC02DJ01_5",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DJ01&&PC02DD01=="5"',
            "INT",
            "逾期R3单月最高逾期（透支）总额",
        ],
        [
            "PC02DS04_5",
            'PCO:PC02:PC02D:PC02DH:[getn(nnfilter(_), 0)]:PC02DS04&&PC02DD01=="5"',
            "INT",
            "逾期R3最长逾期（透支）月数",
        ],
        ["PC02DS01", "PCO:PC02:PC02D:PC02DS01", "INT", "PC02DH 记录数"],
        ["PC02ES01", "PCO:PC02:PC02E:PC02ES01", "INT", "D1管理机构数"],
        ["PC02ES02", "PCO:PC02:PC02E:PC02ES02", "INT", "D1账户数"],
        ["PC02EJ01", "PCO:PC02:PC02E:PC02EJ01", "INT", "D1授信总额"],
        ["PC02EJ02", "PCO:PC02:PC02E:PC02EJ02", "INT", "D1余额"],
        ["PC02EJ03", "PCO:PC02:PC02E:PC02EJ03", "INT", "D1最近6个月平均应还款"],
        ["PC02FS01", "PCO:PC02:PC02F:PC02FS01", "INT", "R4管理机构数"],
        ["PC02FS02", "PCO:PC02:PC02F:PC02FS02", "INT", "R4账户数"],
        ["PC02FJ01", "PCO:PC02:PC02F:PC02FJ01", "INT", "R4授信总额"],
        ["PC02FJ02", "PCO:PC02:PC02F:PC02FJ02", "INT", "R4余额"],
        ["PC02FJ03", "PCO:PC02:PC02F:PC02FJ03", "INT", "R4最近6个月平均应还款"],
        ["PC02GS01", "PCO:PC02:PC02G:PC02GS01", "INT", "R1管理机构数"],
        ["PC02GS02", "PCO:PC02:PC02G:PC02GS02", "INT", "R1账户数"],
        ["PC02GJ01", "PCO:PC02:PC02G:PC02GJ01", "INT", "R1授信总额"],
        ["PC02GJ02", "PCO:PC02:PC02G:PC02GJ02", "INT", "R1余额"],
        ["PC02GJ03", "PCO:PC02:PC02G:PC02GJ03", "INT", "R1最近6个月平均应还款"],
        ["PC02HS01", "PCO:PC02:PC02H:PC02HS01", "INT", "R2管理机构数"],
        ["PC02HS02", "PCO:PC02:PC02H:PC02HS02", "INT", "R2账户数"],
        ["PC02HJ01", "PCO:PC02:PC02H:PC02HJ01", "INT", "R2授信总额"],
        ["PC02HJ02", "PCO:PC02:PC02H:PC02HJ02", "INT", "R2单家行最高授信额"],
        ["PC02HJ03", "PCO:PC02:PC02H:PC02HJ03", "INT", "R2单家行最低授信额"],
        ["PC02HJ04", "PCO:PC02:PC02H:PC02HJ04", "INT", "R2已用额度"],
        ["PC02HJ05", "PCO:PC02:PC02H:PC02HJ05", "INT", "R2最近6个月平均使用额度"],
        ["PC02IS01", "PCO:PC02:PC02I:PC02IS01", "INT", "R3管理机构数"],
        ["PC02IS02", "PCO:PC02:PC02I:PC02IS02", "INT", "R3账户数"],
        ["PC02IJ01", "PCO:PC02:PC02I:PC02IJ01", "INT", "R3授信总额"],
        ["PC02IJ02", "PCO:PC02:PC02I:PC02IJ02", "INT", "R3单家行最高授信额"],
        ["PC02IJ03", "PCO:PC02:PC02I:PC02IJ03", "INT", "R3单家行最低授信额"],
        ["PC02IJ04", "PCO:PC02:PC02I:PC02IJ04", "INT", "R3已用额度"],
        ["PC02IJ05", "PCO:PC02:PC02I:PC02IJ05", "INT", "R3最近6个月平均使用额度"],
    ],
}


# %%
PBOC_REL_ABST = {
    "part": "pboc_rel_abst",
    "desc": "相关还款责任",
    "steps": [
        {
            "content": "PCO:PC02:PC02K:PC02KH:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PC02KD01", "PC02KD01", "VARCHAR(31)", "借款人身份"],
        ["PC02KD02", "PC02KD02", "VARCHAR(31)", "还款责任类型"],
        ["PC02KS02", "PC02KS02", "INT", "账户数"],
        ["PC02KJ01", "PC02KJ01", "INT", "担保责任-还款责任金额"],
        ["PC02KJ02", "PC02KJ02", "INT", "担保责任-账户余额"],
    ],
}


# %%
PBOC_POSTFEE_ABST = {
    "part": "pboc_postfee_abst",
    "desc": "后付费业务概要",
    "steps": [
        {
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
            ]
        }
    ],
    "prikey": ["rid", "certno"],
    "level": 0,
    "fields": [
        [
            "PC030D01_1",
            'PNO:PC03:PC030H:[getn(nnfilter(_), 0)]:PC030D01&&PC030D01=="1"',
            "VARCHAR(31)",
            "后付费业务类型：1-电信业务",
        ],
        [
            "PC030S02_1",
            'PNO:PC03:PC030H:[getn(nnfilter(_), 0)]:PC030S02&&PC030D01=="1"',
            "INT",
            "电信业务欠费账户数",
        ],
        [
            "PC030J01_1",
            'PNO:PC03:PC030H:[getn(nnfilter(_), 0)]:PC030J01&&PC030D01=="1"',
            "INT",
            "电信业务欠费金额",
        ],
        [
            "PC030D01_2",
            'PNO:PC03:PC030H:[getn(nnfilter(_), 0)]:PC030D01&&PC030D01=="2"',
            "VARCHAR(31)",
            "后付费业务类型：2-水电费等公共事业",
        ],
        [
            "PC030S02_2",
            'PNO:PC03:PC030H:[getn(nnfilter(_), 0)]:PC030S02&&PC030D01=="2"',
            "INT",
            "水电费等公共事业欠费账户数",
        ],
        [
            "PC030J01_2",
            'PNO:PC03:PC030H:[getn(nnfilter(_), 0)]:PC030J01&&PC030D01=="2"',
            "INT",
            "水电费等公共事业欠费金额",
        ],
        ["PC030S01", "PNO:PC03:PC030S01", "INT", "后付费业务记录数"],
    ],
}


# %%
PBOC_PUBLIC_ABST = {
    "part": "pboc_public_abst",
    "desc": "公共信息概要",
    "steps": [
        {
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
            ]
        }
    ],
    "prikey": ["rid", "certno"],
    "level": 0,
    "fields": [
        [
            "PC040D01_1",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040D01&&PC040D01=="1"',
            "VARCHAR(31)",
            "公共信息类型：1 欠税信息 ",
        ],
        [
            "PC040S02_1",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040S02&&PC040D01=="1"',
            "INT",
            "欠税信息记录数",
        ],
        [
            "PC040J01_1",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040J01&&PC040D01=="1"',
            "INT",
            "欠税信息涉及金额",
        ],
        [
            "PC040D01_2",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040D01&&PC040D01=="2"',
            "VARCHAR(31)",
            "公共信息类型：2 民事判决信息 ",
        ],
        [
            "PC040S02_2",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040S02&&PC040D01=="2"',
            "INT",
            "民事判决信息记录数",
        ],
        [
            "PC040J01_2",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040J01&&PC040D01=="2"',
            "INT",
            "民事判决信息涉及金额",
        ],
        [
            "PC040D01_3",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040D01&&PC040D01=="3"',
            "VARCHAR(31)",
            "公共信息类型：3 强制执行信息 ",
        ],
        [
            "PC040S02_3",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040S02&&PC040D01=="3"',
            "INT",
            "强制执行信息记录数",
        ],
        [
            "PC040J01_3",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040J01&&PC040D01=="3"',
            "INT",
            "强制执行信息涉及金额",
        ],
        [
            "PC040D01_4",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040D01&&PC040D01=="4"',
            "VARCHAR(31)",
            "公共信息类型：4 行政处罚信息 ",
        ],
        [
            "PC040S02_4",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040S02&&PC040D01=="4"',
            "INT",
            "行政处罚信息记录数",
        ],
        [
            "PC040J01_4",
            'PPO:PC04:PC040H:[getn(nnfilter(_), 0)]:PC040J01&&PC040D01=="4"',
            "INT",
            "行政处罚信息涉及金额",
        ],
        ["PC040S01", "PPO:PC04:PC040S01", "INT", "公共信息记录数"],
    ],
}


# %%
PBOC_INQ_ABST = {
    "part": "pboc_inq_abst",
    "desc": "查询记录概要",
    "steps": [
        {
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
            ]
        }
    ],
    "prikey": ["rid", "certno"],
    "level": 0,
    "fields": [
        ["PC05AR01", "PQO:PC05:PC05A:PC05AR01", "DATE", "上一次查询日期"],
        ["PC05AI01", "PQO:PC05:PC05A:PC05AI01", "VARCHAR(255)", "上一次查询机构代码"],
        ["PC05AD01", "PQO:PC05:PC05A:PC05AD01", "VARCHAR(31)", "上一次查询机构机构类型"],
        ["PC05AQ01", "PQO:PC05:PC05A:PC05AQ01", "VARCHAR(1022)", "上一次查询原因"],
        ["PC05BS01", "PQO:PC05:PC05B:PC05BS01", "INT", "最近1个月内的查询机构数(贷款审批)"],
        ["PC05BS02", "PQO:PC05:PC05B:PC05BS02", "INT", "最近1个月内的查询机构数(信用卡审批)"],
        ["PC05BS03", "PQO:PC05:PC05B:PC05BS03", "INT", "最近1个月内的查询次数(贷款审批)"],
        ["PC05BS04", "PQO:PC05:PC05B:PC05BS04", "INT", "最近1个月内的查询次数(信用卡审批)"],
        ["PC05BS05", "PQO:PC05:PC05B:PC05BS05", "INT", "最近1个月内的查询次数(本人查询)"],
        ["PC05BS06", "PQO:PC05:PC05B:PC05BS06", "INT", "最近2年内的查询次数(贷后管理)"],
        ["PC05BS07", "PQO:PC05:PC05B:PC05BS07", "INT", "最近2年内的查询次数(担保资格审查)"],
        ["PC05BS08", "PQO:PC05:PC05B:PC05BS08", "INT", "最近2年内的查询次数(特约商户实名审查)"],
    ],
}


# %%
PBOC_ACC_INFO = {
    "part": "pboc_acc_info",
    "desc": "分帐户明细",
    "steps": [
        {
            "content": "PDA:PD01:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["accid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "accid"],
    "level": 1,
    "fields": [
        ["PD01AD01", "PD01A:PD01AD01", "VARCHAR(31)", "基本信息_账户类型"],
        ["PD01AD02", "PD01A:PD01AD02", "VARCHAR(31)", "基本信息_业务管理机构类型"],
        ["PD01AD03", "PD01A:PD01AD03", "VARCHAR(31)", "基本信息_业务种类"],
        ["PD01AD04", "PD01A:PD01AD04", "VARCHAR(31)", "基本信息_币种"],
        ["PD01AD05", "PD01A:PD01AD05", "VARCHAR(31)", "基本信息_还款方式"],
        ["PD01AD06", "PD01A:PD01AD06", "VARCHAR(31)", "基本信息_还款频率"],
        ["PD01AD07", "PD01A:PD01AD07", "VARCHAR(31)", "基本信息_担保方式"],
        ["PD01AD08", "PD01A:PD01AD08", "VARCHAR(31)", "基本信息_贷款发放形式"],
        ["PD01AD09", "PD01A:PD01AD09", "VARCHAR(31)", "基本信息_共同借款标志"],
        ["PD01AD10", "PD01A:PD01AD10", "VARCHAR(31)", "基本信息_债权转移时的还款状态"],
        ["PD01AI01", "PD01A:PD01AI01", "VARCHAR(255)", "基本信息_账户编号"],
        ["PD01AI02", "PD01A:PD01AI02", "VARCHAR(255)", "基本信息_业务管理机构代码"],
        ["PD01AI03", "PD01A:PD01AI03", "VARCHAR(255)", "基本信息_账户标识"],
        ["PD01AI04", "PD01A:PD01AI04", "VARCHAR(255)", "基本信息_授信协议编号（PD02AI03）"],
        ["PD01AJ01", "PD01A:PD01AJ01", "INT", "基本信息_借款金额"],
        ["PD01AJ02", "PD01A:PD01AJ02", "INT", "基本信息_账户授信额度"],
        ["PD01AJ03", "PD01A:PD01AJ03", "INT", "基本信息_共享授信额度"],
        ["PD01AR01", "PD01A:PD01AR01", "DATE", "基本信息_账户开立日期"],
        ["PD01AR02", "PD01A:PD01AR02", "DATE", "基本信息_账户到期日期"],
        ["PD01AS01", "PD01A:PD01AS01", "INT", "基本信息_还款期数"],
        ["PD01BD01", "PD01B:PD01BD01", "VARCHAR(31)", "最新表现信息_账户状态"],
        ["PD01BD03", "PD01B:PD01BD03", "VARCHAR(31)", "最新表现信息_五级分类"],
        ["PD01BD04", "PD01B:PD01BD04", "VARCHAR(31)", "最新表现信息_还款状态"],
        ["PD01BJ01", "PD01B:PD01BJ01", "INT", "最新表现信息_账户余额"],
        ["PD01BJ02", "PD01B:PD01BJ02", "INT", "最新表现信息_最近一次还款金额"],
        ["PD01BR01", "PD01B:PD01BR01", "DATE", "最新表现信息_账户关闭日期"],
        ["PD01BR02", "PD01B:PD01BR02", "DATE", "最新表现信息_最近一次还款日期"],
        ["PD01BR03", "PD01B:PD01BR03", "DATE", "最新表现信息_信息报告日期"],
        ["PD01BR04", "PD01B:PD01BR04", "DATE", "最新表现信息_转出月份"],
        ["PD01CD01", "PD01C:PD01CD01", "VARCHAR(31)", "最近一次月度表现信息_账户状态"],
        ["PD01CD02", "PD01C:PD01CD02", "VARCHAR(31)", "最近一次月度表现信息_五级分类"],
        ["PD01CJ01", "PD01C:PD01CJ01", "INT", "最近一次月度表现信息_账户余额"],
        ["PD01CJ02", "PD01C:PD01CJ02", "INT", "最近一次月度表现信息_已用额度"],
        ["PD01CJ03", "PD01C:PD01CJ03", "INT", "最近一次月度表现信息_未出单的大额专项分期余额"],
        ["PD01CJ04", "PD01C:PD01CJ04", "INT", "最近一次月度表现信息_本月应还款"],
        ["PD01CJ05", "PD01C:PD01CJ05", "INT", "最近一次月度表现信息_本月实还款"],
        ["PD01CJ06", "PD01C:PD01CJ06", "INT", "最近一次月度表现信息_当前逾期总额"],
        ["PD01CJ07", "PD01C:PD01CJ07", "INT", "最近一次月度表现信息_逾期31-60天未还本金"],
        ["PD01CJ08", "PD01C:PD01CJ08", "INT", "最近一次月度表现信息_逾期61-90天未还本金"],
        ["PD01CJ09", "PD01C:PD01CJ09", "INT", "最近一次月度表现信息_逾期91-180天未还本金"],
        ["PD01CJ10", "PD01C:PD01CJ10", "INT", "最近一次月度表现信息_逾期180天以上未还本金"],
        ["PD01CJ11", "PD01C:PD01CJ11", "INT", "最近一次月度表现信息_透支180天以上未付余额"],
        ["PD01CJ12", "PD01C:PD01CJ12", "INT", "最近一次月度表现信息_最近6个月平均使用额度"],
        ["PD01CJ13", "PD01C:PD01CJ13", "INT", "最近一次月度表现信息_最近6个月平均透支余额"],
        ["PD01CJ14", "PD01C:PD01CJ14", "INT", "最近一次月度表现信息_最大使用额度"],
        ["PD01CJ15", "PD01C:PD01CJ15", "INT", "最近一次月度表现信息_最大透支余额"],
        ["PD01CR01", "PD01C:PD01CR01", "DATE", "最近一次月度表现信息_月份"],
        ["PD01CR02", "PD01C:PD01CR02", "DATE", "最近一次月度表现信息_本月应还款日"],
        ["PD01CR03", "PD01C:PD01CR03", "DATE", "最近一次月度表现信息_最近一次还款日期"],
        ["PD01CR04", "PD01C:PD01CR04", "DATE", "最近一次月度表现信息_信息报告日期"],
        ["PD01CS01", "PD01C:PD01CS01", "INT", "最近一次月度表现信息_剩余还款期数"],
        ["PD01CS02", "PD01C:PD01CS02", "INT", "最近一次月度表现信息_当期逾期期数"],
        ["PD01DR01", "PD01D:PD01DR01", "DATE", "近24个月 - 起始年月"],
        ["PD01DR02", "PD01D:PD01DR02", "DATE", "近24个月 - 截至年月"],
        ["PD01ER01", "PD01E:PD01ER01", "DATE", "近60个月 - 起始年月"],
        ["PD01ER02", "PD01E:PD01ER02", "DATE", "近60个月 - 截至年月"],
        ["PD01ES01", "PD01E:PD01ES01", "INT", "近60个月 - 记录数"],
        ["PD01FS01", "PD01F:PD01FS01", "INT", "特殊交易记录数"],
        ["PD01GS01", "PD01G:PD01GS01", "INT", "特殊事件记录数"],
        ["PD01HS01", "PD01H:PD01HS01", "INT", "大额专项分期记录数"],
        ["PD01ZH", "PD01Z:PD01ZH", "TEXT", "标注或声明内容"],
        ["PD01ZS01", "PD01Z:PD01ZS01", "INT", "PD01ZH 中记录数"],
    ],
}


# %%
PBOC_ACC_REPAY_24_MONTHLY = {
    "part": "pboc_acc_repay_24_monthly",
    "desc": "分帐户明细_近24个月还款",
    "steps": [
        {
            "content": "PDA:PD01:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["accid", "RANGEINDEX"],
            ],
        },
        {"content": "PD01D:PD01DH:[_]", "key": [["monid", "RANGEINDEX"]]},
    ],
    "prikey": ["rid", "certno", "accid", "monid"],
    "level": 2,
    "fields": [
        ["PD01DR03", "PD01DR03", "DATE", "近24个月_月份"],
        ["PD01DD01", "PD01DD01", "VARCHAR(31)", "近24个月_还款状态"],
    ],
}


# %%
PBOC_ACC_REPAY_60_MONTHLY = {
    "part": "pboc_acc_repay_60_monthly",
    "desc": "分帐户明细_近60个月还款",
    "steps": [
        {
            "content": "PDA:PD01:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["accid", "RANGEINDEX"],
            ],
        },
        {"content": "PD01E:PD01EH:[_]", "key": [["monid", "RANGEINDEX"]]},
    ],
    "prikey": ["rid", "certno", "accid", "monid"],
    "level": 2,
    "fields": [
        ["PD01ER03", "PD01ER03", "DATE", "月份"],
        ["PD01ED01", "PD01ED01", "VARCHAR(31)", "还款状态"],
        ["PD01EJ01", "PD01EJ01", "INT", "逾期（透支）总额"],
    ],
}


# %%
PBOC_ACC_SPECIAL_TRANS = {
    "part": "pboc_acc_special_trans",
    "desc": "分帐户明细_特殊交易",
    "steps": [
        {
            "content": "PDA:PD01:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["accid", "RANGEINDEX"],
            ],
        },
        {"content": "PD01F:PD01FH:[_]", "key": [["acctid", "RANGEINDEX"]]},
    ],
    "prikey": ["rid", "certno", "accid", "acctid"],
    "level": 2,
    "fields": [
        ["PD01FD01", "PD01FD01", "VARCHAR(31)", "特殊交易类型"],
        ["PD01FR01", "PD01FR01", "DATE", "特殊交易发生日期"],
        ["PD01FS02", "PD01FS02", "INT", "到期日期变更月数"],
        ["PD01FJ01", "PD01FJ01", "INT", "特殊交易发生金额"],
        ["PD01FQ01", "PD01FQ01", "VARCHAR(1022)", "特殊交易明细记录"],
    ],
}


# %%
PBOC_ACC_SPECIAL_ACCD = {
    "part": "pboc_acc_special_accd",
    "desc": "分账户明细_特殊事件",
    "steps": [
        {
            "content": "PDA:PD01:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["accid", "RANGEINDEX"],
            ],
        },
        {"content": "PD01G:PD01GH:[_]", "key": [["acctid", "RANGEINDEX"]]},
    ],
    "prikey": ["rid", "certno", "accid", "acctid"],
    "level": 2,
    "fields": [
        ["PD01GR01", "PD01GR01", "DATE", "特殊事件发生月份"],
        ["PD01GD01", "PD01GD01", "VARCHAR(31)", "特殊事件类型"],
    ],
}


# %%
PBOC_ACC_SPECIAL_INSTS = {
    "part": "pboc_acc_special_insts",
    "desc": "分帐户明细_大额专项分期",
    "steps": [
        {
            "content": "PDA:PD01:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["accid", "RANGEINDEX"],
            ],
        },
        {"content": "PD01H:PD01HH:[_]", "key": [["acctid", "RANGEINDEX"]]},
    ],
    "prikey": ["rid", "certno", "accid", "acctid"],
    "level": 2,
    "fields": [
        ["PD01HJ01", "PD01HJ01", "INT", "大额专项分期额度"],
        ["PD01HJ02", "PD01HJ02", "INT", "已用分期额度"],
        ["PD01HR01", "PD01HR01", "DATE", "分期额度生效日期"],
        ["PD01HR02", "PD01HR02", "DATE", "分期额度到期日期"],
    ],
}


# %%
PBOC_CREDIT_INFO = {
    "part": "pboc_credit_info",
    "desc": "授信协议明细",
    "steps": [
        {
            "content": "PCA:PD02:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PD02AD01", "PD02A:PD02AD01", "VARCHAR(31)", "授信协议-业务管理机构类型"],
        ["PD02AD02", "PD02A:PD02AD02", "VARCHAR(31)", "授信额度用途"],
        ["PD02AD03", "PD02A:PD02AD03", "VARCHAR(31)", "币种"],
        ["PD02AD04", "PD02A:PD02AD04", "VARCHAR(31)", "授信协议状态（全空）"],
        ["PD02AI01", "PD02A:PD02AI01", "VARCHAR(255)", "授信协议编号"],
        ["PD02AI02", "PD02A:PD02AI02", "VARCHAR(255)", "业务管理机构"],
        ["PD02AI03", "PD02A:PD02AI03", "VARCHAR(255)", "授信协议标识（PD01AI04）"],
        ["PD02AI04", "PD02A:PD02AI04", "VARCHAR(255)", "授信限额编号"],
        ["PD02AJ01", "PD02A:PD02AJ01", "INT", "授信额度"],
        ["PD02AJ03", "PD02A:PD02AJ03", "INT", "授信限额"],
        ["PD02AJ04", "PD02A:PD02AJ04", "INT", "已用额度"],
        ["PD02AR01", "PD02A:PD02AR01", "DATE", "生效日期"],
        ["PD02AR02", "PD02A:PD02AR02", "DATE", "到期日期"],
        ["PD02ZH", "PD02Z:PD02ZH", "TEXT", "标注或声明类型"],
        ["PD02ZS01", "PD02Z:PD02ZS01", "INT", "PD02ZH 中记录数"],
    ],
}


# %%
PBOC_REL_INFO = {
    "part": "pboc_rel_info",
    "desc": "相关还款责任",
    "steps": [
        {
            "content": "PCR:PD03:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PD03AD01", "PD03A:PD03AD01", "VARCHAR(31)", "相关还款责任-业务管理机构类型"],
        ["PD03AD02", "PD03A:PD03AD02", "VARCHAR(31)", "业务种类"],
        ["PD03AD03", "PD03A:PD03AD03", "VARCHAR(31)", "相关还款责任人类型"],
        ["PD03AD04", "PD03A:PD03AD04", "VARCHAR(31)", "币种"],
        ["PD03AD05", "PD03A:PD03AD05", "VARCHAR(31)", "五级分类"],
        ["PD03AD06", "PD03A:PD03AD06", "VARCHAR(31)", "账户类型"],
        ["PD03AD07", "PD03A:PD03AD07", "VARCHAR(31)", "还款状态"],
        ["PD03AD08", "PD03A:PD03AD08", "VARCHAR(31)", "主借款人身份类别"],
        ["PD03AJ01", "PD03A:PD03AJ01", "INT", "相关还款责任金额"],
        ["PD03AJ02", "PD03A:PD03AJ02", "INT", "账户金额"],
        ["PD03AQ01", "PD03A:PD03AQ01", "VARCHAR(1022)", "业务管理机构"],
        ["PD03AQ02", "PD03A:PD03AQ02", "VARCHAR(1022)", "保证合同编号"],
        ["PD03AR01", "PD03A:PD03AR01", "DATE", "账户开立日期"],
        ["PD03AR02", "PD03A:PD03AR02", "DATE", "账户到期日期"],
        ["PD03AR03", "PD03A:PD03AR03", "DATE", "信息报告日期"],
        ["PD03AS01", "PD03A:PD03AS01", "INT", "逾期月数"],
        ["PD03ZH", "PD03Z:PD03ZH", "TEXT", "标注或声明类型"],
        ["PD03ZS01", "PD03Z:PD03ZS01", "INT", "PD03ZH 中记录数"],
    ],
}


# %%
PBOC_POSTFEE_INFO = {
    "part": "pboc_postfee_info",
    "desc": "非信贷交易明细",
    "steps": [
        {
            "content": "PND:PE01:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PE01AD01", "PE01A:PE01AD01", "VARCHAR(31)", "后付费账户类型"],
        ["PE01AD02", "PE01A:PE01AD02", "VARCHAR(31)", "业务类型"],
        ["PE01AD03", "PE01A:PE01AD03", "VARCHAR(31)", "当前缴费状态"],
        ["PE01AJ01", "PE01A:PE01AJ01", "INT", "当前欠费金额"],
        ["PE01AQ01", "PE01A:PE01AQ01", "VARCHAR(1022)", "机构名称"],
        ["PE01AQ02", "PE01A:PE01AQ02", "VARCHAR(1022)", "最近24个月缴费记录"],
        ["PE01AR01", "PE01A:PE01AR01", "DATE", "业务开通日期"],
        ["PE01AR02", "PE01A:PE01AR02", "DATE", "记账年月"],
        ["PE01ZH", "PE01Z:PE01ZH", "TEXT", "标注或声明类型"],
        ["PE01ZS01", "PE01Z:PE01ZS01", "INT", "PE01ZH 中记录数"],
    ],
}


# %%
PBOC_TAXS = {
    "part": "pboc_taxs",
    "desc": "欠税记录",
    "steps": [
        {
            "content": "POT:PF01:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PF01AJ01", "PF01A:PF01AJ01", "INT", "欠税总额"],
        ["PF01AQ01", "PF01A:PF01AQ01", "VARCHAR(1022)", "主管税务机关"],
        ["PF01AR01", "PF01A:PF01AR01", "DATE", "欠税统计日期"],
        ["PF01ZH", "PF01Z:PF01ZH", "TEXT", "标注或声明类型"],
        ["PF01ZS01", "PF01Z:PF01ZS01", "INT", "PF01ZH 记录数"],
    ],
}


# %%
PBOC_LAWSUIT = {
    "part": "pboc_lawsuit",
    "desc": "民事判决记录",
    "steps": [
        {
            "content": "PCJ:PF02:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PF02AD01", "PF02A:PF02AD01", "VARCHAR(31)", "结案方式"],
        ["PF02AJ01", "PF02A:PF02AJ01", "INT", "诉讼标的金额"],
        ["PF02AQ01", "PF02A:PF02AQ01", "VARCHAR(1022)", "立案法院"],
        ["PF02AQ02", "PF02A:PF02AQ02", "VARCHAR(1022)", "案由"],
        ["PF02AQ03", "PF02A:PF02AQ03", "VARCHAR(1022)", "判决、调解结果"],
        ["PF02AQ04", "PF02A:PF02AQ04", "VARCHAR(1022)", "诉讼标的"],
        ["PF02AR01", "PF02A:PF02AR01", "DATE", "立案日期"],
        ["PF02AR02", "PF02A:PF02AR02", "DATE", "判决、调节生效日期"],
        ["PF02ZH", "PF02Z:PF02ZH", "TEXT", "标注或声明类型"],
        ["PF02ZS01", "PF02Z:PF02ZS01", "INT", "PF02ZH 中记录数"],
    ],
}


# %%
PBOC_ENFORCEMENT = {
    "part": "pboc_enforcement",
    "desc": "强制执行记录",
    "steps": [
        {
            "content": "PCE:PF03:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PF03AD01", "PF03A:PF03AD01", "VARCHAR(31)", "结案方式"],
        ["PF03AJ01", "PF03A:PF03AJ01", "INT", "申请执行标的金额"],
        ["PF03AJ02", "PF03A:PF03AJ02", "INT", "已执行标的金额"],
        ["PF03AQ01", "PF03A:PF03AQ01", "VARCHAR(1022)", "执行法院"],
        ["PF03AQ02", "PF03A:PF03AQ02", "VARCHAR(1022)", "执行案由"],
        ["PF03AQ03", "PF03A:PF03AQ03", "VARCHAR(1022)", "案件状态"],
        ["PF03AQ04", "PF03A:PF03AQ04", "VARCHAR(1022)", "申请执行标的"],
        ["PF03AQ05", "PF03A:PF03AQ05", "VARCHAR(1022)", "已执行标的"],
        ["PF03AR01", "PF03A:PF03AR01", "DATE", "立案日期"],
        ["PF03AR02", "PF03A:PF03AR02", "DATE", "结案日期"],
        ["PF03ZH", "PF03Z:PF03ZH", "TEXT", "标注或声明类型"],
        ["PF03ZS01", "PF03Z:PF03ZS01", "INT", "PF03ZH 中记录数"],
    ],
}


# %%
PBOC_GOV_PUNISHMENT = {
    "part": "pboc_gov_punishment",
    "desc": "行政处罚记录",
    "steps": [
        {
            "content": "PAP:PF04:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PF04AJ01", "PF04A:PF04AJ01", "INT", "处罚金额"],
        ["PF04AQ01", "PF04A:PF04AQ01", "VARCHAR(1022)", "处罚机构"],
        ["PF04AQ02", "PF04A:PF04AQ02", "VARCHAR(1022)", "处罚内容"],
        ["PF04AQ03", "PF04A:PF04AQ03", "VARCHAR(1022)", "行政复议结果"],
        ["PF04AR01", "PF04A:PF04AR01", "DATE", "处罚生效日期"],
        ["PF04AR02", "PF04A:PF04AR02", "DATE", "处罚截至日期"],
        ["PF04ZH", "PF04Z:PF04ZH", "TEXT", "标注或声明类型"],
        ["PF04ZS01", "PF04Z:PF04ZS01", "INT", "PF04ZH 中记录数"],
    ],
}


# %%
PBOC_HOUSING_FUND = {
    "part": "pboc_housing_fund",
    "desc": "住房公积金参缴记录",
    "steps": [
        {
            "content": "PHF:PF05:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PF05AD01", "PF05A:PF05AD01", "VARCHAR(31)", "缴费状态"],
        ["PF05AJ01", "PF05A:PF05AJ01", "INT", "月存缴额"],
        ["PF05AQ01", "PF05A:PF05AQ01", "VARCHAR(1022)", "参缴地"],
        ["PF05AQ02", "PF05A:PF05AQ02", "INT", "个人缴存比例"],
        ["PF05AQ03", "PF05A:PF05AQ03", "INT", "单位缴存比例"],
        ["PF05AQ04", "PF05A:PF05AQ04", "VARCHAR(1022)", "缴费单位"],
        ["PF05AR01", "PF05A:PF05AR01", "DATE", "参缴日期"],
        ["PF05AR02", "PF05A:PF05AR02", "DATE", "初缴月份"],
        ["PF05AR03", "PF05A:PF05AR03", "DATE", "缴至月份"],
        ["PF05AR04", "PF05A:PF05AR04", "DATE", "信息更新日期"],
        ["PF05ZH", "PF05Z:PF05ZH", "TEXT", "标注或声明类型"],
        ["PF05ZS01", "PF05Z:PF05ZS01", "INT", "PF05ZH 中记录数"],
    ],
}


# %%
PBOC_SUB_ALLOWANCE = {
    "part": "pboc_sub_allowance",
    "desc": "低保救助记录",
    "steps": [
        {
            "content": "PBS:PF06:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PF06AD01", "PF06A:PF06AD01", "VARCHAR(31)", "人员类别"],
        ["PF06AQ01", "PF06A:PF06AQ01", "VARCHAR(1022)", "所在地"],
        ["PF06AQ02", "PF06A:PF06AQ02", "VARCHAR(1022)", "工作单位"],
        ["PF06AQ03", "PF06A:PF06AQ03", "VARCHAR(1022)", "家庭月收入"],
        ["PF06AR01", "PF06A:PF06AR01", "DATE", "申请日期"],
        ["PF06AR02", "PF06A:PF06AR02", "DATE", "批准日期"],
        ["PF06AR03", "PF06A:PF06AR03", "DATE", "信息更新日期"],
        ["PF06ZH", "PF06Z:PF06ZH", "TEXT", "标注或声明类型"],
        ["PF06ZS01", "PF06Z:PF06ZS01", "INT", "PF06ZH 中记录数"],
    ],
}


# %%
PBOC_PRO_CERT = {
    "part": "pboc_pro_cert",
    "desc": "执业资格记录",
    "steps": [
        {
            "content": "PPQ:PF07:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PF07AD01", "PF07A:PF07AD01", "VARCHAR(31)", "等级"],
        ["PF07AD02", "PF07A:PF07AD02", "VARCHAR(1022)", "机构所在地"],
        ["PF07AQ01", "PF07A:PF07AQ01", "VARCHAR(1022)", "职业资格名称"],
        ["PF07AQ02", "PF07A:PF07AQ02", "VARCHAR(1022)", "颁发机构"],
        ["PF07AR01", "PF07A:PF07AR01", "DATE", "获得年月"],
        ["PF07AR02", "PF07A:PF07AR02", "DATE", "到期年月"],
        ["PF07AR03", "PF07A:PF07AR03", "DATE", "吊销年月"],
        ["PF07ZH", "PF07Z:PF07ZH", "TEXT", "标注或声明类型"],
        ["PF07ZS01", "PF07Z:PF07ZS01", "INT", "PF07ZH 中记录数"],
    ],
}


# %%
PBOC_GOV_AWARD = {
    "part": "pboc_gov_award",
    "desc": "行政奖励记录",
    "steps": [
        {
            "content": "PAH:PF08:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PF08AQ01", "PF08A:PF08AQ01", "VARCHAR(1022)", "奖励机构"],
        ["PF08AQ02", "PF08A:PF08AQ02", "VARCHAR(1022)", "奖励内容"],
        ["PF08AR01", "PF08A:PF08AR01", "DATE", "生效年月"],
        ["PF08AR02", "PF08A:PF08AR02", "DATE", "截至年月"],
        ["PF08ZH", "PF08Z:PF08ZH", "TEXT", "标注或声明类型"],
        ["PF08ZS01", "PF08Z:PF08ZS01", "INT", "PF08ZH 中记录数"],
    ],
}


# %%
PBOC_INQ_REC = {
    "part": "pboc_inq_rec",
    "desc": "查询记录",
    "steps": [
        {
            "content": "POQ:PH01:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PH010D01", "PH010D01", "VARCHAR(31)", "查询机构类型"],
        ["PH010Q02", "PH010Q02", "VARCHAR(1022)", "查询机构"],
        ["PH010Q03", "PH010Q03", "VARCHAR(1022)", "查询原因"],
        ["PH010R01", "PH010R01", "DATE", "查询日期"],
    ],
}


# %%
PBOC_ANNOTATION = {
    "part": "pboc_annotation",
    "desc": "标注声明",
    "steps": [
        {
            "content": "POS:PG01:[_]",
            "key": [
                ["rid", "PRH:PA01:PA01A:PA01AI01"],
                ["certno", "PRH:PA01:PA01B:PA01BI01"],
                ["recid", "RANGEINDEX"],
            ],
        }
    ],
    "prikey": ["rid", "certno", "recid"],
    "level": 1,
    "fields": [
        ["PG010D01", "PG010D01", "VARCHAR(31)", "标注对象类型"],
        ["PG010D02", "PG010D02", "VARCHAR(31)", "标注对象标识"],
        ["PG010H", "PG010H", "TEXT", "标注或声明类型"],
        ["PG010S01", "PG010S01", "INT", "PG010H 中记录数量"],
    ],
}


# %%
PBOC_PARTS = {
    PBOC_BASIC_INFO["part"]: PBOC_BASIC_INFO,
    PBOC_MOBILE["part"]: PBOC_MOBILE,
    PBOC_ADDRESS["part"]: PBOC_ADDRESS,
    PBOC_COMPANY["part"]: PBOC_COMPANY,
    PBOC_SCORE["part"]: PBOC_SCORE,
    PBOC_BIZ_ABST["part"]: PBOC_BIZ_ABST,
    PBOC_CACC_ABST["part"]: PBOC_CACC_ABST,
    PBOC_DUM_ABST["part"]: PBOC_DUM_ABST,
    PBOC_DRACC_ABST["part"]: PBOC_DRACC_ABST,
    PBOC_REL_ABST["part"]: PBOC_REL_ABST,
    PBOC_POSTFEE_ABST["part"]: PBOC_POSTFEE_ABST,
    PBOC_PUBLIC_ABST["part"]: PBOC_PUBLIC_ABST,
    PBOC_INQ_ABST["part"]: PBOC_INQ_ABST,
    PBOC_ACC_INFO["part"]: PBOC_ACC_INFO,
    PBOC_ACC_REPAY_24_MONTHLY["part"]: PBOC_ACC_REPAY_24_MONTHLY,
    PBOC_ACC_REPAY_60_MONTHLY["part"]: PBOC_ACC_REPAY_60_MONTHLY,
    PBOC_ACC_SPECIAL_TRANS["part"]: PBOC_ACC_SPECIAL_TRANS,
    PBOC_ACC_SPECIAL_ACCD["part"]: PBOC_ACC_SPECIAL_ACCD,
    PBOC_ACC_SPECIAL_INSTS["part"]: PBOC_ACC_SPECIAL_INSTS,
    PBOC_CREDIT_INFO["part"]: PBOC_CREDIT_INFO,
    PBOC_REL_INFO["part"]: PBOC_REL_INFO,
    PBOC_POSTFEE_INFO["part"]: PBOC_POSTFEE_INFO,
    PBOC_TAXS["part"]: PBOC_TAXS,
    PBOC_LAWSUIT["part"]: PBOC_LAWSUIT,
    PBOC_ENFORCEMENT["part"]: PBOC_ENFORCEMENT,
    PBOC_GOV_PUNISHMENT["part"]: PBOC_GOV_PUNISHMENT,
    PBOC_HOUSING_FUND["part"]: PBOC_HOUSING_FUND,
    PBOC_SUB_ALLOWANCE["part"]: PBOC_SUB_ALLOWANCE,
    PBOC_PRO_CERT["part"]: PBOC_PRO_CERT,
    PBOC_GOV_AWARD["part"]: PBOC_GOV_AWARD,
    PBOC_INQ_REC["part"]: PBOC_INQ_REC,
    PBOC_ANNOTATION["part"]: PBOC_ANNOTATION,
}


# %%
def df_flat_confs():
    import pandas as pd
    pboc_parts = []
    pboc_fields = []
    for val in PBOC_PARTS.values():
        part_one = {
            "part": val["part"],
            "level": val["level"],
            "prikey": val["prikey"],
            "steps": val["steps"],
            "desc": val["desc"],
        }
        pboc_parts.append(part_one)
        pboc_fields.extend([[val["part"], *ele] for ele in val["fields"]])
    pboc_parts = pd.DataFrame(pboc_parts)
    pboc_fields = pd.DataFrame.from_records(
        pboc_fields, columns=["part", "field", "step", "dtype", "desc"])

    return pboc_parts, pboc_fields
