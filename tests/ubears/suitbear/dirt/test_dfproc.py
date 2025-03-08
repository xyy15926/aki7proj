#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_dfproc.py
#   Author: xyy15926
#   Created: 2025-02-17 14:42:36
#   Updated: 2025-03-08 19:20:01
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ubears.suitbear.dirt import dfproc
    reload(dfproc)

import numpy as np
import pandas as pd
from ubears.suitbear.dirt.dfproc import (
    fixup_df, trans_arr, trans_df,
    drop_fields,
)


# %%
def mock_data(row_n=100):
    seed = 777
    np.random.seed(seed)

    id_ = np.arange(row_n)
    np.random.shuffle(id_)
    range_int = np.arange(1, row_n + 1)
    range_int_half = np.concatenate([np.arange(1, row_n // 2 + 1),
                                     np.arange(1, row_n // 2 + 1)])

    int_01 = np.random.randint(0, 1, row_n)
    int_02 = np.random.randint(0, 2, row_n)
    int_03 = np.random.randint(0, 3, row_n)
    int_05 = np.random.randint(0, 5, row_n)
    int_10 = np.random.randint(0, 10, row_n)

    int_str = np.random.choice(list("123456789"), row_n)
    mixed_int_str = np.random.choice(list("123456789ab"), row_n)

    float_10 = np.random.randint(1, 10, row_n)
    float_10 = float_10 + np.random.random(row_n).round(1)

    # Float with NaNs.
    nan_float_sers = {}
    flags = np.arange(row_n)
    for rr in [0.1, 0.2, 0.5, 0.8, 0.9, 1]:
        np.random.shuffle(flags)
        float_val = np.random.randint(1, 10, row_n).astype(np.float64)
        rrn = int(rr * row_n)
        float_val[flags[:rrn]] = np.nan
        nan_float_sers[f"nan_float_{rrn:03}"] = float_val

    # Float with flat-1s.
    flat_float_sers = {}
    flags = np.arange(row_n)
    for rr in [0.1, 0.2, 0.5, 0.8, 0.9, 1]:
        np.random.shuffle(flags)
        float_val = np.random.randint(1, 10, row_n).astype(np.float64)
        rrn = int(rr * row_n)
        float_val[flags[:rrn]] = 101
        flat_float_sers[f"flat_float_{rrn:03}"] = float_val

    label = np.random.randint(0, 2, row_n)

    return pd.DataFrame.from_dict({
        "id": id_,
        "range_int": range_int,
        "range_int_half": range_int_half,
        "int_01": int_01,
        "int_02": int_02,
        "int_03": int_03,
        "int_05": int_05,
        "int_10": int_10,
        "int_str": int_str,
        "mixed_int_str": mixed_int_str,
        "float_10": float_10,
        **nan_float_sers,
        **flat_float_sers,
    }), pd.Series(label)


# %%
def keep_last(subdf: pd.DataFrame):
    return subdf.iloc[-1]


@pytest.mark.filterwarnings("ignore: divide by zero encountered")
@pytest.mark.filterwarnings("ignore: invalid value encountered")
def test_fixup_df():
    data, label = mock_data()
    confs = {
        "sort_keys": {
            "id": True,
        },
        "uni_keys": ["int_05", "int_10"],
        "uni_keep": keep_last,
        "na_thresh": 0.55,
        "flat_thresh": 0.85,
        "fillna": {
            "__NUM__": -999999,
            "__CAT__": "ZNA",
        },
        "pcorr_thresh": 1,
    }
    ret = fixup_df(data, confs)

    assert ret.shape[0] < data.shape[0]
    assert ret.shape[1] < data.shape[1]
    assert not np.any(ret.duplicated(subset=["int_05", "int_10"]))
    assert ret["id"].is_monotonic_increasing
    assert np.all(ret.notna())
    for coln in ret.columns:
        if coln.startswith("nan_float"):
            assert np.any(ret[coln] == -999999)
            nar = int(coln[-3:])
            assert nar < 55
        if coln.startswith("flat_float"):
            nar = int(coln[-3:])
            assert nar < 90

    reret = fixup_df(ret, confs)
    # Index has been reset.
    assert np.all(ret.values == reret.values)


# %%
@pytest.mark.filterwarnings("ignore: divide by zero encountered")
@pytest.mark.filterwarnings("ignore: invalid value encountered")
def test_trans_arr():
    data, label = mock_data()

    # Numeric Series.
    ser = data["float_10"]
    num_conf_1 = [
        {
            "ttype": "binize",
            "nbin": 5,
        }
    ]
    retp1 = trans_arr(ser, label, num_conf_1)
    assert len(np.unique(retp1)) == 5
    num_conf_2 = [
        {
            "ttype": "woe"
        }
    ]
    retp2 = trans_arr(retp1, label, num_conf_2)
    assert len(np.unique(retp2)) == 5
    num_conf_3 = [
        {
            "ttype": "map",
            "ref": lambda x: x + 1,
        }
    ]
    retp3 = trans_arr(retp2, label, num_conf_3)
    p23gap = retp3 - retp2
    assert np.all(p23gap[~np.isnan(p23gap)] == 1)

    # Just like pipline?
    num_conf = num_conf_1 + num_conf_2
    ret2 = trans_arr(ser, label, num_conf)
    assert len(np.unique(ret2)) == 5
    assert np.all(np.isclose(retp2, ret2, equal_nan=True))

    # Non-Numeric Series.
    ser = data["nan_float_010"]
    nbin = [-999999, 0, 3, 6, 10]
    cat_conf_1 = [
        {
            "ttype": "ordinize",
            "nafill": -999999,
        }, {
            "ttype": "binize",
            "nbin": nbin,
        }, {
            "ttype": "map",
            "ref": dict(zip(range(len(nbin)), range(1, len(nbin) + 1))),
        }
    ]
    ret3 = trans_arr(ser, label, cat_conf_1)
    assert len(np.unique(ret3)) == len(nbin) - 1
    assert np.all(ret3 >= 1)
    cat_conf_2 = [
        {
            "ttype": "woe",
        }
    ]
    ret4 = trans_arr(ret3, label, cat_conf_2)
    assert len(np.unique(ret4)) == len(nbin) - 1


# %%
@pytest.mark.filterwarnings("ignore: divide by zero encountered")
@pytest.mark.filterwarnings("ignore: invalid value encountered")
def test_trans_df():
    data, label = mock_data()
    num_cols = data.select_dtypes(include=np.number).columns
    cat_cols = data.select_dtypes(exclude=np.number).columns
    exclude = ["id", "__LABEL__"]

    nbin = [-999999, 0, 3, 6, 10]
    check_conf = {
        "sort_keys": {
            "id": True,
        },
        "uni_keys": ["int_05", "int_10"],
        "uni_keep": keep_last,
        "na_thresh": 0.55,
        "flat_thresh": 0.85,
        "fillna": {
            "__NUM__": -999999,
            "__CAT__": "ZNA",
        },
        "pcorr_thresh": 1,
    }
    trans_conf = {
        "__NUM__": [
            {
                "ttype": "binize",
                "nbin": 5,
            }, {
                "ttype": "woe",
            }, {
                "ttype": "map",
                "ref": lambda x: x - 1,
            },
        ],
        "__CAT__": [
            {
                "ttype": "ordinize",
                "nafill": -999999,
            }, {
                "ttype": "map",
                "ref": {
                    "0": 1,
                    "1": 2,
                    "2": 3,
                },
                "default": 99,
            }, {
                "ttype": "binize",
                "nbin": nbin,
            }, {
                "ttype": "woe",
            }
        ],
    }

    data["__LABEL__"] = label
    checked = fixup_df(data, check_conf, exclude=exclude)
    label = checked["__LABEL__"]
    ret = trans_df(checked, label, trans_conf, exclude=exclude)
    for col in ret.columns:
        if col in exclude:
            continue
        if col in num_cols:
            # Some may stop early in `tree_cut`.
            assert len(np.unique(ret[col])) <= 5
        elif col in cat_cols:
            assert len(np.unique(ret[col])) <= len(nbin) - 1


# %%
@pytest.mark.filterwarnings("ignore: divide by zero encountered")
@pytest.mark.filterwarnings("ignore: invalid value encountered")
def test_drop_fields():
    data, label = mock_data()
    exclude = ["id"]
    data["float_10_c1"] = data["float_10"]
    assert "int_05" in data.columns

    dropc_conf_1 = {
        "manual": ["int_05"],
        "pcorr_thresh": 1,
    }
    ret = drop_fields(data, label, dropc_conf_1, exclude=exclude)
    assert ret.shape[1] == data.shape[1] - 2
    assert "int_05" not in ret
    for eele in exclude:
        assert eele in ret

    dropc_conf_2 = {
        "iv_thresh": 0.1,
    }
    ret2 = drop_fields(ret, label, dropc_conf_2, exclude=exclude)
    assert ret2.shape[1] < ret.shape[1]
    for eele in exclude:
        assert eele in ret2
