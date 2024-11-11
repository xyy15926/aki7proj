#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_ex2df.py
#   Author: xyy15926
#   Created: 2024-11-11 14:17:07
#   Updated: 2024-11-11 14:22:21
#   Description:
# ---------------------------------------------------------

# %%
import pytest

if __name__ == "__main__":
    from importlib import reload
    from flagbear.str2 import fliper
    from modsbear.dflater import ex2df, exenv
    reload(fliper)
    reload(ex2df)
    reload(exenv)

import numpy as np
import pandas as pd
import os
import json

from flagbear.slp.finer import get_assets_path
from flagbear.str2.fliper import extract_field, rebuild_dict
from modsbear.dflater.ex2df import rebuild_rec2df, compress_hierarchy, flat_records

PBOC_CONFS = get_assets_path() / "pboc_settings"
PBOC_JSON = PBOC_CONFS / "pboc_utf8.json"


# %%
def pboc_src():
    pboc = open(PBOC_JSON, "r").read()
    pboc2 = pboc.replace("2019101617463675115707", "2019101617463675115708")
    src = pd.Series({"xfy": pboc, "xfy2": pboc2})

    return src


# %%
def test_rebuild_rec2df_basic():
    src = pboc_src()
    rec = src.iloc[0]

    # Test basic function.
    val_rules = [
        ["pboc_acc_info", "PDA:PD01:[_]"],
    ]
    index_rules = [
        ["rid", "PRH:PA01:PA01A:PA01AI01"],
        ["certno", "PRH:PA01:PA01B:PA01BI01"],
    ]
    # Test `explode` flag.
    nrec = rebuild_rec2df(rec, val_rules, index_rules, explode=False)
    assert len(nrec) == 1
    assert np.all(nrec.columns == [i[0] for i in val_rules])

    nrec = rebuild_rec2df(rec, val_rules, index_rules, explode=True)
    assert len(nrec) > 1
    assert np.all(nrec.columns == [i[0] for i in val_rules])

    val_rules = [
        ["pboc_basic_info_A", "PRH:PA01"],
        ["pboc_basic_info_B", "PIM:PB01"],
    ]
    index_rules = [
        ["rid", "PRH:PA01:PA01A:PA01AI01"],
        ["certno", "PRH:PA01:PA01B:PA01BI01"],
    ]
    # Test multiple value extraction rules.
    nrec = rebuild_rec2df(rec, val_rules, index_rules, explode=True)
    assert len(nrec) == 1
    assert np.all(nrec.columns == [i[0] for i in val_rules])

    # Test no index extaction rules passed.
    nrec = rebuild_rec2df(rec, val_rules, [], explode=True)
    assert np.all(nrec.columns == [i[0] for i in val_rules])

    # Test no value extaction rules passed.
    nrec = rebuild_rec2df(rec, None, index_rules, explode=True)
    assert np.all(nrec.index.names == [i[0] for i in index_rules])

    # Test neither value rules nor index rules passed.
    nrec = rebuild_rec2df(rec, [], [], explode=False)
    assert nrec.empty
    nrec = rebuild_rec2df(rec, None, [], explode=False)
    assert nrec.empty
    nrec = rebuild_rec2df(rec, [], None, explode=False)
    assert nrec.empty
    nrec = rebuild_rec2df(rec, None, None, explode=False)
    assert nrec.empty


def test_rebuild_rec2df_range_index():
    src = pboc_src()
    rec = src.iloc[0]

    val_rules = [
        ["pboc_acc_info", "PDA:PD01:[_]"],
    ]
    index_rules = [
        ["rid", "PRH:PA01:PA01A:PA01AI01"],
        ["certno", "PRH:PA01:PA01B:PA01BI01"],
        ["PD01AI01", "PDA:PD01:[_]:PD01A:PD01AI01"]
    ]
    nrec = rebuild_rec2df(rec, val_rules, index_rules, explode=True)
    assert len(nrec) >= 1
    assert np.all(nrec.columns == [i[0] for i in val_rules])
    assert np.all(nrec.index.names == [i[0] for i in index_rules])

    nrec = rebuild_rec2df(rec, val_rules, index_rules, explode=True, range_index="ridx")
    assert len(nrec) >= 1
    assert np.all(nrec.columns == [i[0] for i in val_rules])
    assert np.all(nrec.index.names == [i[0] for i in index_rules] + ["ridx"])


# %%
def test_rebuild_rec2df_explode():
    src = pboc_src()
    rec = src.iloc[0]

    # Test `explode` shouldn't be set when extractions don't share the same
    # length.
    val_rules = [
        ["pboc_acc_info", "PDA:PD01:[_]"],
        ["pboc_basic_info_A", "PRH:PA01"],
        ["pboc_basic_info_B", "PIM:PB01"],
    ]
    index_rules = [
        ["rid", "PRH:PA01:PA01A:PA01AI01"],
        ["certno", "PRH:PA01:PA01B:PA01BI01"],
    ]
    nrec = rebuild_rec2df(rec, val_rules, index_rules, explode=False)
    assert len(nrec) == 1
    assert np.all(nrec.columns == [i[0] for i in val_rules])

    with pytest.raises(ValueError):
        nrec = rebuild_rec2df(rec, val_rules, index_rules, explode=True)


# %%
def test_rebuild_rec2df_null_fields():
    src = pboc_src()
    rec = src.iloc[0]

    # Null fields test.
    val_rules = [
        ["pboc_acc_None", "a:b:c"],
        ["pboc_basic_None", "c:d"],
    ]
    index_rules = [
        ["rid", "a:b:c"],
        ["certno", "c:d"],
    ]
    nrec = rebuild_rec2df(rec, val_rules, index_rules, explode=False)
    assert len(nrec) == 1
    assert np.all(nrec.columns == [i[0] for i in val_rules])


# %%
def test_rebuild_rec2df_check_dtype():
    rec = {
        "int": ["", ""],
        "varchar": ["", ""],
        "date": ["2024-05", "abc"]
    }
    val_rules = [
        ("int", None, "int", "INT", np.nan),
        ("varchar", "varchar", "VARCHAR(255)"),
        ("date", None, "date", "DATE", np.datetime64("NaT")),
        ("null_date", None, "null_date", "DATE", np.datetime64("NaT")),
    ]
    nrec = rebuild_rec2df(rec, val_rules, explode=True)
    assert len(nrec) == 2

    val_rules_element_wise = [
        ("int", None, "int:[_]", "INT", np.nan),
        ("varchar", "varchar:[_]", "VARCHAR(255)"),
        ("date", None, "date:[_]", "DATE", np.datetime64("NaT")),
        ("null_date", None, "null_date:[_]", "DATE", np.datetime64("NaT")),
    ]
    element_wise_nrec = rebuild_rec2df(rec, val_rules_element_wise,
                                       explode=True)

    assert np.any(nrec.dtypes == element_wise_nrec.dtypes)
    assert np.any(nrec.dtypes != element_wise_nrec.dtypes)


# %%
def test_compress_hierarchy():
    src = pboc_src()

    acc_info_part = [
        {
            "content": "PDA:PD01:[_]:PD01A",
            "key": [
                ("rid", "PRH:PA01:PA01A:PA01AI01"),
                ("certno", "PRH:PA01:PA01B:PA01BI01"),
                ("accid", "PDA:PD01:[_]:PD01A:PD01AI01"),
            ]
        }
    ]
    acc_info_psrc = compress_hierarchy(src, acc_info_part)
    assert len(acc_info_psrc) > len(src)
    assert acc_info_psrc.index.nlevels == src.index.nlevels + 2 + 1

    repay_60m_part = [
        {
            "content": "PDA:PD01:[_]:PD01E",
            "key": [
                ("rid", "PRH:PA01:PA01A:PA01AI01"),
                ("certno", "PRH:PA01:PA01B:PA01BI01"),
                ("accid", "PDA:PD01:[_]:PD01A:PD01AI01"),
            ]
        },{
            "content": "PD01EH:[_]",
        }
    ]
    repay_60m_psrc = compress_hierarchy(src, repay_60m_part)
    assert repay_60m_psrc.index.nlevels == src.index.nlevels + 3 + 1
    assert len(repay_60m_psrc) > len(acc_info_psrc)

    repay_60m_part_v2 = [
        {"content": "PDA:PD01:[_]"},
        {"content": "PD01E:PD01EH:[_]"},
    ]
    repay_60m_psrc_v2 = compress_hierarchy(src, repay_60m_part_v2)
    assert repay_60m_psrc_v2.index.nlevels == src.index.nlevels + 1 + 1
    assert np.all(repay_60m_psrc.values == repay_60m_psrc_v2.values)

    return acc_info_psrc, repay_60m_psrc


# %%
def test_compress_hierarchy_range_idx():
    src = pboc_src()
    # `RANGEINDEX` in `idkey_0` represents used the RangeIndex as the index.
    repay_60m_part = [
        {
            "content": "PDA:PD01:[_]:PD01E",
            "key": [
                ("rid", "PRH:PA01:PA01A:PA01AI01"),
                ("certno", "PRH:PA01:PA01B:PA01BI01"),
                ("accid", "RANGEINDEX"),
            ]
        },{
            "content": "PD01EH:[_]",
        }
    ]
    repay_60m_psrc = compress_hierarchy(src, repay_60m_part)
    assert repay_60m_psrc.index.nlevels == src.index.nlevels + 3 + 1


# %%
def test_compress_hierarchy_null_vals():
    src = pboc_src()
    repay_60m_part = [
        {
            "content": "PDA:PD01:[_]:PD01E",
            "key": [
                ("rid", "PRH:PA01:PA01A:PA01AI01"),
                ("certno", "PRH:PA01:PA01B:PA01BI01"),
                ("accid", "PDA:PD01:[_]:PD01A:PD01AI01"),
            ]
        },{
            "content": "PD01AH:[_]",
        }
    ]
    repay_60m_psrc = compress_hierarchy(src, repay_60m_part)
    assert repay_60m_psrc.empty


def test_compress_hierarchy_empty_list():
    src = pd.Series({"a": []})
    conf = [
        {"content": "a"},
        {"content": "d"},
        {"content": "e"},
    ]
    nrec = compress_hierarchy(src, conf)
    assert nrec.empty


# %%
def test_flat_record():
    acc_info_psrc, repay_60m_psrc = test_compress_hierarchy()

    acc_info_fields = [
        ["PD01AD01", "PD01AD01", "VARCHAR(31)", "基本信息_账户类型"],
        ["PD01AD02", "PD01AD02", "VARCHAR(31)", "基本信息_业务管理机构类型"],
        ["PD01AD03", "PD01AD03", "VARCHAR(31)", "基本信息_业务种类"],
        ["PD01AD04", "PD01AD04", "VARCHAR(31)", "基本信息_币种"],
        ["PD01AI01", "PD01AI01", "VARCHAR(255)", "基本信息_账户编号"],
        ["PD01AR01", "PD01AR01", "DATE", "基本信息_开日日期"],
        ["PD01AR02", "PD01AR02", "DATE", "基本信息_到期日期"],
    ]

    tuple2_fields = [ele[:2] for ele in acc_info_fields]
    acc_info_vals = flat_records(acc_info_psrc, tuple2_fields)
    assert np.all(acc_info_vals.index.names == acc_info_psrc.index.names)
    assert len(acc_info_vals) == len(acc_info_psrc)
    assert np.all(acc_info_vals.dtypes == "object")

    tuple3_fields = [ele[:3] for ele in acc_info_fields]
    acc_info_vals = flat_records(acc_info_psrc, tuple3_fields)
    assert np.all(acc_info_vals.index.names == acc_info_psrc.index.names)
    assert len(acc_info_vals) == len(acc_info_psrc)
    assert np.sum(acc_info_vals.dtypes != "object") == 1

    mixed_fields = []
    for key, step, dtype, desc in acc_info_fields:
        if dtype == "DATE":
            mixed_fields.append([key, None, step, dtype, np.datetime64("nat")])
        else:
            mixed_fields.append([key, step])
    acc_info_vals = flat_records(acc_info_psrc, mixed_fields)
    assert np.all(acc_info_vals.index.names == acc_info_psrc.index.names)
    assert len(acc_info_vals) == len(acc_info_psrc)
    assert np.sum(acc_info_vals.dtypes != "object") == 2

    repay_60m_fields = [
        ("PD01ER03", "PD01ER03", "date", "月份"),
        ("PD01ED01", "PD01ED01", "varchar(31)", "还款状态"),
        ("PD01EJ01", "PD01EJ01", "int", "逾期（透支）总额"),
    ]
    tuple2_fields = [ele[:2] for ele in repay_60m_fields]
    repay_60m_vals = flat_records(repay_60m_psrc, tuple2_fields, drop_rid=False)
    assert np.all(repay_60m_vals.index.names[:-1] == repay_60m_psrc.index.names)
    assert len(repay_60m_vals) == len(repay_60m_psrc)

    return acc_info_vals, repay_60m_vals
