#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_dflog.py
#   Author: xyy15926
#   Created: 2024-01-18 20:28:51
#   Updated: 2024-11-11 15:12:26
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark

if __name__ == "__main__":
    from importlib import reload
    from ringbear import metrics
    from ringbear import sortable
    from modsbear.spanner import dflog
    reload(metrics)
    reload(sortable)
    reload(dflog)

import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import contingency

from ringbear.metrics import cal_woes
from ringbear.sortable import tree_cut
from modsbear.spanner.dflog import serlog, serdiffm


# %%
def test_serlog():
    ser = pd.Series([99, 2, 3, "a", 99, None, float('nan')])
    label = pd.Series(np.ones(len(ser), dtype=np.int_))
    uni_df, col_D = serlog(ser, label)
    unis = pd.unique(ser)
    assert len(uni_df) == len(unis) - 1


def test_serdiffm():
    ser = pd.Series([99, 2, 3, "a", 99, None, float("nan"), float("nan")])
    codes = pd.Series(pd.factorize(ser)[0])
    mapper = serdiffm(ser, codes)
    assert np.all(ser.map(mapper).fillna(-1) == codes)

    mapper = serdiffm(ser, ser)
    assert not np.all(mapper.index == mapper.values)

    # Check how to handle `np.nan` in numeric interval ocassion.
    ser = pd.Series([99, 2, 3, 4, 99, 2, float("nan")])
    codes = pd.Series(pd.factorize(ser)[0])
    itvl_mapper = serdiffm(ser, codes, to_interval=True)
    assert max(itvl_mapper.index.levels[1]) == ser.max()
    assert np.all(np.isnan(itvl_mapper.index[-1]))

    # Check how to handle only one unique element.
    ser = pd.Series([99, 99, 99])
    codes = pd.Series(pd.factorize(ser)[0])
    itvl_mapper = serdiffm(ser, codes, to_interval=True)
    assert max(itvl_mapper.index.levels[1]) == ser.max()
