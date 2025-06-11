#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_ex2df.py
#   Author: xyy15926
#   Created: 2024-11-11 14:17:07
#   Updated: 2025-01-21 21:49:43
#   Description:
# ---------------------------------------------------------

# %%
import pytest

if __name__ == "__main__":
    from importlib import reload
    from ubears.flagbear.str2 import dtyper, fliper
    from ubears.modsbear.dflater import ex2df, exenv
    reload(dtyper)
    reload(fliper)
    reload(ex2df)
    reload(exenv)

import numpy as np
import pandas as pd
import os
import json

from ubears.flagbear.str2.fliper import extract_field, rebuild_dict
from ubears.modsbear.dflater.ex2df import rebuild_rec2df, compress_hierarchy, flat_records


# %%
def pboc_rec():
    src = {
        "PRH": {
            "PA01": {
                "PA01A": {
                    "PA01AI01": "2019101617463675115707",
                    "PA01AR01": "2019-10-16T17:46:36"
                },
                "PA01B": {
                    "PA01BD01": "10",
                    "PA01BD02": "22",
                    "PA01BI01": "622926198501293785",
                    "PA01BI02": "A10311000H0001",
                    "PA01BQ01": "王小二"

                },
                "PA01C": None,
                "PA01D": None,
                "PA01E": {
                    "PA01ES01": "1"
                }
            }
        },
        "PDA": {
            "PD01": [
                {
                    "PD01A": {
                        "PD01AD01": "D1",
                        "PD01AD02": "12",
                        "PD01AD03": "12",
                        "PD01AD04": "JPY",
                        "PD01AI01": "92719",
                        "PD01AJ01": "10000",
                        "PD01AJ03": None,
                        "PD01AR01": "2013-01-10",
                        "PD01AR02": None,
                    },
                    "PD01B": {
                        "PD01BD01": "3",
                        "PD01BD04": "",
                        "PD01BJ01": "0",
                        "PD01BJ02": "",
                        "PD01BR01": "2018-06-20",
                        "PD01BR02": "--",
                    },
                    "PD01C": None,
                    "PD01D": {
                        "PD01DH": [
                            {
                                "PD01DD01": "#",
                                "PD01DR03": "2016-08"
                            },
                            {
                                "PD01DD01": "#",
                                "PD01DR03": "2016-07"
                            }
                        ],
                        "PD01DR01": "2016-07",
                        "PD01DR02": "2018-06"
                    },
                    "PD01E": {
                        "PD01EH": [
                            {
                                "PD01ED01": "#",
                                "PD01EJ01": "",
                                "PD01ER03": "2014-12"
                            },
                            {
                                "PD01ED01": "#",
                                "PD01EJ01": "",
                                "PD01ER03": "2014-11"
                            },
                            {
                                "PD01ED01": "#",
                                "PD01EJ01": "",
                                "PD01ER03": "2014-10"
                            }
                        ],
                        "PD01ER01": "2014-11",
                        "PD01ER02": "2018-06",
                        "PD01ES01": "44"
                    },
                    "PD01F": None,
                    "PD01G": None,
                    "PD01H": None,
                    "PD01Z": None,
                },
                {
                    "PD01A": {
                        "PD01AD01": "R4",
                        "PD01AD02": "12",
                        "PD01AD03": "92",
                        "PD01AD04": "CNY",
                        "PD01AI01": "92720",
                        "PD01AJ01": "20000",
                        "PD01AJ03": None,
                        "PD01AR01": "2013-01-10",
                        "PD01AR02": None,
                    },
                    "PD01B": {
                        "PD01BD01": "3",
                        "PD01BD04": "",
                        "PD01BJ01": "0",
                        "PD01BJ02": "",
                        "PD01BR01": "2018-06-20",
                        "PD01BR02": "--",
                    },
                    "PD01C": None,
                    "PD01D": {
                        "PD01DH": [
                            {
                                "PD01DD01": "#",
                                "PD01DR03": "2016-08"
                            },
                            {
                                "PD01DD01": "#",
                                "PD01DR03": "2016-07"
                            }
                        ],
                        "PD01DR01": "2016-07",
                        "PD01DR02": "2018-06"
                    },
                    "PD01E": None,
                    "PD01F": None,
                    "PD01G": None,
                    "PD01H": None,
                    "PD01Z": None,
                },
            ],
        }
    }

    return src


# %%
def test_rebuild_rec2df_basic():
    rec = pboc_rec()

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
    rec = pboc_rec()

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
    rec = pboc_rec()

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
    rec = pboc_rec()

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
    assert np.all(nrec.index.names == [i[0] for i in index_rules])


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
    rec = pboc_rec()
    rec2 = rec.copy()
    rec2["PRH"]["PA01"]["PA01A"]["PA01AI01"] = "2019101617463675115708"
    src = pd.Series([rec, rec2])

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


# %%
def cal_compress_hierarchy():
    rec = pboc_rec()
    rec2 = rec.copy()
    rec2["PRH"]["PA01"]["PA01A"]["PA01AI01"] = "2019101617463675115708"
    src = pd.Series([rec, rec2])

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

    return acc_info_psrc, repay_60m_psrc


# %%
def test_compress_hierarchy_range_idx():
    rec = pboc_rec()
    rec2 = rec.copy()
    rec2["PRH"]["PA01"]["PA01A"]["PA01AI01"] = "2019101617463675115708"
    src = pd.Series([rec, rec2])

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
    rec = pboc_rec()
    rec2 = rec.copy()
    rec2["PRH"]["PA01"]["PA01A"]["PA01AI01"] = "2019101617463675115708"
    src = pd.Series([rec, rec2])

    # Null content.
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
    assert len(repay_60m_psrc.index.names) == 1 + 3 + 1


# %%
def test_flat_record():
    acc_info_psrc, repay_60m_psrc = cal_compress_hierarchy()

    acc_info_fields = [
        ["PD01AD01", "PD01AD01", "VARCHAR(31)", "基本信息_账户类型"],
        ["PD01AD02", "PD01AD02", "VARCHAR(31)", "基本信息_业务管理机构类型"],
        ["PD01AD03", "PD01AD03", "VARCHAR(31)", "基本信息_业务种类"],
        ["PD01AD04", "PD01AD04", "VARCHAR(31)", "基本信息_币种"],
        ["PD01AI01", "PD01AI01", "VARCHAR(255)", "基本信息_账户编号"],
        ["PD01AR01", "PD01AR01", "DATE", "基本信息_开立日期"],
        ["PD01AR02", "PD01AR02", "DATE", "基本信息_到期日期"],
    ]

    # Flat with 2-Tuple confs.
    tuple2_fields = [ele[:2] for ele in acc_info_fields]
    acc_info_vals = flat_records(acc_info_psrc, tuple2_fields)
    assert np.all(acc_info_vals.index.names == acc_info_psrc.index.names)
    assert len(acc_info_vals) == len(acc_info_psrc)
    assert np.all(acc_info_vals.dtypes == "object")

    # Flat with 2-Tuple null confs.
    tuple2_fields = [[a, b + "ccc"] for a, b, *c in acc_info_fields]
    acc_info_vals = flat_records(acc_info_psrc, tuple2_fields)
    assert np.all(acc_info_vals.index.names == acc_info_psrc.index.names)
    assert len(acc_info_vals) == len(acc_info_psrc)
    assert np.all(acc_info_vals.dtypes == "object")

    # Flat with 2-Tuple confs with null data source.
    tuple2_fields = [ele[:2] for ele in acc_info_fields]
    acc_info_psrc_null = pd.Series(index=acc_info_psrc.index, dtype=object)
    acc_info_vals = flat_records(acc_info_psrc_null, tuple2_fields)
    assert np.all(acc_info_vals.index.names == acc_info_psrc.index.names)
    assert len(acc_info_vals) == len(acc_info_psrc)
    assert np.all(acc_info_vals.dtypes == "object")

    # Flat with 3-Tuple confs.
    tuple3_fields = [ele[:3] for ele in acc_info_fields]
    acc_info_vals = flat_records(acc_info_psrc, tuple3_fields)
    assert np.all(acc_info_vals.index.names == acc_info_psrc.index.names)
    assert len(acc_info_vals) == len(acc_info_psrc)
    assert np.sum(acc_info_vals.dtypes != "object") == 1

    # Flat with 5-Tuple confs.
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

    # Flat with 6-Tuple confs.
    mixed_fields = []
    for key, step, dtype, desc in acc_info_fields:
        if dtype == "DATE":
            mixed_fields.append([key, None, step, dtype, None, True])
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
