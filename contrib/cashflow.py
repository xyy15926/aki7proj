#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: cashflow.py
#   Author: xyy15926
#   Created: 2024-08-25 15:28:52
#   Updated: 2024-08-25 22:04:30
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple
import logging

import numpy as np
import pandas as pd

from suitbear.finer import get_assets_path, get_tmp_path

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
COUNTER_PARTY_KEYWORDS = {
    "航旅出行": [
        r".*航空.*",
        r".*酒店.*",
        r".*中国铁路网络有限公司.*",
        r".*12306.*",
    ],
    "自驾出行": [
        r".*代驾.*",
        r".*高速.*",
        r".*ETC.*",
        r".*石油.*",
        r".*石化.*",
        r".*加油.*",
        r".*停车.*",
    ],
    "公共交通": [
        r".*地铁.*",
        r".*公交.*",
    ],
    "打车": [
        r".*滴滴.*",
        r".*T3出行.*",
        r".*高德出行.*",
        r".*曹操出行.*",
    ],
    "家居消费": [
        r".*燃气.*",
        r".*供电.*",
        r".*供水.*",
    ],
    "保险": [
        r".*保险.*公司.*",
    ],
    "投资理财": [
        r".*购买理财通.*",
        r".*理财通赎回.*",
    ],
}

RISK_TRANSACTION_AMOUNT = [
    888, 888.88, 8888, 8888.88, 666, 666.66, 6666, 6666.66, 520, 1314
]

WEXIN_TRANSACTION_TYPE = {
    "tf_in"         : '(交易类型 == "转账") and (收支 == "收入")',
    "tf_out"        : '(交易类型 == "转账") and (收支 == "支出")',
    "qrc_in"        : '交易类型 == "二维码收款"',
    "qrc_out"       : '交易类型 == "扫二维码付款"',
    "hb_in"         : 're_match(交易类型, ".*微信红包.*") and (收支 == "收入")',
    "hb_out"        : 're_match(交易类型, ".*微信红包.*") and (收支 == "支出")',
    "group_in"      : '(交易类型 == "群收款") and (收支 == "收入")',
    "group_out"     : '(交易类型 == "群收款") and (收支 == "支出")',
    "comsu"         : '交易类型 == "商户消费"',
    "outof"         : '交易类型 == "其他"',
    "credit_card"   : '交易类型 == "信用卡还款"',
    "change_out"    : '交易类型 == "零钱提现"',
}

WEXIN_TRANSACTION_METHOD = {
    "credit_card"   : 're_match(交易方式, ".*信用卡.*")',
    "debit_card"    : 're_match(交易方式, ".*储蓄卡.*")',
    "change"        : '(交易方式 == "零钱") or (交易方式 == "零钱通")',
    "biz_acc"       : '交易方式 == "经营账户"',
}

ALIPAY_TRANSACTION_TYPE = {
    "comsu"         : '商家订单号 != ""',
    "tf_out"        : '(商家订单号 == "") and (收支 == "支出")',
    "tf_in"         : '(商家订单号 == "") and (收支 == "收入")',
    "qrc_out"       : 're_match(商品说明, ".*付款.*") and (收支 == "支出")',
    "qrc_in"        : 're_match(商品说明, ".*收款.*") and (收支 == "收入")',
    "outof"         : '交易类型 == "不计收支"',
}

