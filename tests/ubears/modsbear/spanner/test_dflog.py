#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_dflog.py
#   Author: xyy15926
#   Created: 2024-01-18 20:28:51
#   Updated: 2025-02-14 22:06:03
#   Description:
# ---------------------------------------------------------

# %%
import pytest

if __name__ == "__main__":
    from importlib import reload
    from ubears.ringbear import metrics
    from ubears.ringbear import sortable
    from ubears.modsbear.spanner import dflog
    reload(metrics)
    reload(sortable)
    reload(dflog)

import numpy as np
import pandas as pd
from ubears.modsbear.spanner.dflog import serdesc, serdiffm, dfdesc, dfdiffm
from ubears.modsbear.spanner.dflog import ProcessLogger


# %%
@pytest.mark.filterwarnings("ignore: divide by zero encountered")
def test_serdesc():
    # FreqR, WOE and Lift will all be calculated.
    ser = pd.Series([99, 2, 3, 2, 3, 99])
    label = np.array([1, 0, 0, 1, 0, 1])
    factor_log, seq_log = serdesc(ser, label)
    uni_label = pd.unique(label)
    assert len(factor_log.columns) == len(uni_label) + 9
    assert len(seq_log) == 9

    # Only Freq will be calculated if label is not passed.
    factor_log, seq_log = serdesc(ser)
    assert len(factor_log.columns) == 4
    assert len(seq_log) == 0

    # Uncomparable Series will be factorized first.
    # And Pearson correlation won't be calculated.
    ser = pd.Series(["a", 2, 3, 2, 3, "a"])
    label = np.array([1, 0, 0, 1, 0, 1])
    factor_log, seq_log = serdesc(ser, label)
    uni_label = pd.unique(label)
    assert len(factor_log.columns) == len(uni_label) + 9
    assert len(seq_log) == 8


# %%
@pytest.mark.filterwarnings("ignore: divide by zero encountered")
def test_serdiffm():
    # Mapper returned by `serdiffm` could be used as the factorization.
    ser = pd.Series([99, 2, 3, "a", 99, None, float("nan"), float("nan")])
    codes = pd.Series(pd.factorize(ser)[0])
    mapper = serdiffm(ser, codes)
    assert np.all(ser.map(mapper).fillna(-1) == codes)

    # None will be returned if nothing changed.
    mapper = serdiffm(ser, ser)
    assert mapper is None

    # Check how to handle `np.nan` in numeric interval ocassion.
    ser = pd.Series([99, 2, 3, 4, 99, 2, float("nan")])
    codes = pd.factorize(ser)[0]
    itvl_mapper = serdiffm(ser, codes, to_interval=True)
    assert max(itvl_mapper.index.levels[1]) == ser.max()
    assert np.all(np.isnan(itvl_mapper.index[-1]))

    ser = pd.Series([99, 2, 3, 4, 99, 2, float("nan")])
    codes = ser // 10
    itvl_mapper = serdiffm(ser, codes, to_interval=True)
    assert max(itvl_mapper.index.levels[1]) == ser.max()
    assert np.all(np.isnan(itvl_mapper.index[-1]))

    # Check how to handle only one unique element.
    ser = pd.Series([99, 99, 99])
    codes = pd.Series(pd.factorize(ser)[0])
    itvl_mapper = serdiffm(ser, codes, to_interval=True)
    assert max(itvl_mapper.index.levels[1]) == ser.max()

    ser = pd.Series([99, 99, 99, np.nan])
    codes = pd.Series(pd.factorize(ser)[0])
    itvl_mapper = serdiffm(ser, codes, to_interval=True)
    assert max(itvl_mapper.index.levels[1]) == ser.max()


# %%
@pytest.mark.filterwarnings("ignore: divide by zero encountered")
def test_dfdesc():
    df = pd.DataFrame({"a": [99, 2, 3, 2, 3, 99],
                       "b": ["a", 2, 3, 2, 3, "a"]})
    label = np.array([1, 0, 0, 1, 0, 1])
    fdf, sdf = dfdesc(df, label)
    assert np.all(fdf.loc["a"].values == fdf.loc["b"].values)
    assert np.all(sdf.iloc[0, :-1] == sdf.iloc[1, :-1])


# %%
@pytest.mark.filterwarnings("ignore: divide by zero encountered")
def test_dfdiffm():
    dfo = pd.DataFrame({"a": [99, 2, 3, 4, 99, 2, float("nan")],
                        "b": [99, 2, 3, 4, 99, 2, float("nan")],
                        "c": [99] * 7,
                        "d": [99, 2, "a", 4, 99, 2, float("nan")]})
    dfn = pd.DataFrame({"a": [0, 1, 2, 3, 0, 1, -1],
                        "b": [9, 0, 0, 0, 9, 0, np.nan],
                        "c": [0] * 7,
                        "d": [0, 0, 0, 1, 0, 0, 1]})
    cat_df, num_df = dfdiffm(dfo, dfn, True)

    assert len(cat_df.index.levels[0]) == 1
    assert cat_df.index.nlevels == 2
    assert len(num_df.index.levels[0]) == 3
    assert num_df.index.nlevels == 3

    # `num_df` is empty with `to_interval` uset.
    cat_df, num_df = dfdiffm(dfo, dfn)
    assert num_df.empty

    # Empty mapping DataFrame of Series with mixed-dtype.
    dfo = dfo.iloc[:, :-1]
    dfn = dfn.iloc[:, :-1]
    cat_df, num_df = dfdiffm(dfo, dfn, True)

    assert cat_df.empty
    assert len(num_df.index.levels[0]) == 3
    assert num_df.index.nlevels == 3


# %%
@pytest.mark.filterwarnings("ignore: divide by zero encountered")
@pytest.mark.filterwarnings("ignore: invalid value encountered")
def test_process_logger():
    logger = ProcessLogger()
    ldf = logger.proc_logs
    dfo = pd.DataFrame({"a": [99, 2, 3, 4, 99, 2, float("nan")],
                        "b": [99, 2, 3, 4, 99, 2, float("nan")],
                        "c": [99] * 7,
                        "d": [99, 2, "a", 4, 99, 2, float("nan")]})
    dfn = pd.DataFrame({"a": [0, 1, 2, 3, 0, 1, -1],
                        "b": [9, 0, 0, 0, 9, 0, np.nan],
                        "c": [0] * 7,
                        "d": [0, 0, 0, 1, 0, 0, 1]})
    label = np.array([1, 0, 0, 1, 0, 1, 0])

    logger.vallog(dfo, label, ltag="ori")
    assert len(ldf) == 3
    assert list(ldf.keys()) == ["fdesc_ori_1", "cdesc_ori_1", "pcorr_ori_1"]

    logger.maplog(dfo, dfn, ltag="trans")
    assert len(ldf) == 5
    assert list(ldf.keys())[3:] == ["catmap_trans_1", "nummap_trans_1"]

    logger.vallog(dfn, label, ltag="new")
    assert len(ldf) == 8
    assert list(ldf.keys())[5:] == ["fdesc_new_1", "cdesc_new_1", "pcorr_new_1"]

    logger.vallog(dfn, label, ltag="new")
    assert len(ldf) == 11
    assert list(ldf.keys())[8:] == ["fdesc_new_2", "cdesc_new_2", "pcorr_new_2"]
