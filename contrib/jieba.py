#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: jieba.py
#   Author: xyy15926
#   Created: 2024-07-24 19:05:15
#   Updated: 2025-02-14 10:04:52
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import List, Tuple

import pandas as pd
import jieba
from jieba import posseg as pseg
from suitbear.finer import get_assets_path

logging.basicConfig(
    # format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    format="%(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
# Jieba Ref: <https://github.com/fxsjy/jieba?tab=readme-ov-file>
# Govern Region Data: <https://github.com/xiangyuecn/AreaCity-JsSpider-StatsGov>
# China Area Mysql: <https://github.com/kakuilan/china_area_mysql>
# Geocoding Ref: <https://github.com/casuallyName/Geocoding>
# 词性：
# - a 形容词
#     - ad 副形词
#     - ag 形容词性语素
#     - an 名形词
# - b 区别词
# - c 连词
# - d 副词
#     - df
#     - dg 副语素
# - e 叹词
# - f 方位词
# - g 语素
# - h 前接成分
# - i 成语
# - j 简称略称
# - k 后接成分
# - l 习用语
# - m 数词
#     - mg
#     - mq 数量词
# - n 名词
#     - ng 名词性语素
#     - nr 人名
#     - nrfg
#     - nrt
#     - ns 地名
#     - nt 机构团体名
#     - nz 其他专名
# - o 拟声词
# - p 介词
# - q 量词
# - r 代词
#     - rg 代词性语素
#     - rr 人称代词
#     - rz 指示代词
# - s 处所词
# - t 时间词
#     - tg 时语素
# - u 助词
#     - ud 结构助词 得
#     - ug 时态助词
#     - uj 结构助词 的
#     - ul 时态助词 了
#     - uv 结构助词 地
#     - uz 时态助词 着
# - v 动词
#     - vd 副动词
#     - vg 动词性语素
#     - vi 不及物动词
#     - vn 名动词
#     - vq
# - x 非语素词
# - y 语气词
# - z 状态词
#     - zg
def jieba_toker():
    # Generate customed user dict from government regions infos.
    reg_df = pd.read_csv(get_assets_path() / "govern_region_level4.csv")
    reg_df["pos"] = "ns"
    reg_names = reg_df[["name", "deep", "pos"]].drop_duplicates("name").copy()
    reg_names["deep"] = 1000 // (reg_names["deep"] + 1) ** 2
    reg_exts = reg_df[["ext_name", "deep", "pos"]].drop_duplicates("ext_name").copy()
    reg_exts["deep"] = 1500 // (reg_exts["deep"] + 1) ** 2
    reg_exts.set_axis(["name", "deep", "pos"], axis=1, inplace=True)
    reg_names = (pd.concat([reg_names, reg_exts])
                 .to_csv(get_assets_path() / "govern_region_names.txt",
                         sep=" ",
                         columns=None,
                         header=None))

    # Init new customed Tokenizer.
    # `jiebe.dt` is the default Tokenizer, which is delegated by `jiebe`.
    toker = jieba.Tokenizer()
    # Load user dict.
    toker.add_word("熊风扬", tag="nr")
    toker.add_word("aki7", tag="nz")
    toker.load_userdict(open(get_assets_path() / "govern_region_names.txt",
                             encoding="utf8"))

    return toker


# %%
def jieba_cut():
    toker = jieba_toker()
    # Return generator.
    sent = "中国北京市深圳市开平区熊风扬"

    # Precise mode.
    tok_gen = toker.cut(sent)
    # Return-all mode.
    tok_gen_all = toker.cut(sent, cut_all=True)
    toks = toker.lcut(sent)
    toks_all = toker.lcut(sent, cut_all=True)

    assert list(tok_gen) == toks
    assert list(tok_gen_all) == toks_all
    assert len(toks) < len(toks_all)

    # For search engine, like return-all mode, but words are in different order.
    tok_gen_se = toker.cut_for_search(sent)
    tok_se = toker.lcut_for_search(sent)

    assert list(tok_gen_se) == tok_se
    assert sorted(tok_se) == sorted(toks_all)


# %%
def jieba_pos_cut():
    toker = jieba_toker()
    ptoker = pseg.POSTokenizer(toker)
    sent = "中国北京市深圳市开平区熊风扬"

    pairs = ptoker.lcut(sent)
    words = toker.lcut(sent)
    assert [ele.word for ele in pairs] == words
