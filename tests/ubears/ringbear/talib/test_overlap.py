#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_overlap.py
#   Author: xyy15926
#   Created: 2024-11-20 08:46:58
#   Updated: 2024-11-22 08:54:26
#   Description:
# ---------------------------------------------------------

# %%
import pytest
try:
    import talib as ta
except ImportError:
    pytestmark = pytest.mark.skip(reason="TA-Lib uninstalled.")
import numpy as np
if __name__ == "__main__":
    from importlib import reload
    from ubears.ringbear.talib import overlap
    reload(overlap)

from ubears.ringbear.talib.overlap import (
    ma, sma, wma, trima, trima_yao,
    mavp,
    ema, dema, tema, t3,
    kama,
    midprice, midpoint,
    bbands,
    sar, sar_ext
)


# %%
def mock_data():
    dt = np.random.rand(100, 3)
    dt.sort(axis=1)
    return dt[:, 0], dt[:, 1], dt[:, 2]


# %%
def test_ma():
    low, close, high = mock_data()
    vals = ma(close, 30, 0)
    ta_vals = sma(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))

    vals = ma(close, 30, 2)
    ta_vals = wma(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))

    vals = ma(close, 30, 3)
    ta_vals = trima(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))


# %%
def test_sma():
    low, close, high = mock_data()
    vals = sma(close, 30)
    ta_vals = ta.SMA(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))


# %%
def test_wma():
    low, close, high = mock_data()
    vals = wma(close, 30)
    ta_vals = ta.WMA(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))



# %%
def test_trima():
    low, close, high = mock_data()
    vals = trima(close, 30)
    ta_vals = ta.TRIMA(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))

    vals = trima_yao(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))


# %%
def test_mavp():
    low, close, high = mock_data()
    periods = np.random.randint(5, 10, 100)
    vals = mavp(close, periods)
    ta_vals = ta.MAVP(close, periods.astype(float))
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))


# %%
def test_ema():
    low, close, high = mock_data()
    vals = ema(close, 30)
    ta_vals = ta.EMA(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))


# %%
def test_dema():
    low, close, high = mock_data()
    vals = dema(close, 30)
    ta_vals = ta.DEMA(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))


# %%
def test_tema():
    low, close, high = mock_data()
    vals = tema(close, 30)
    ta_vals = ta.TEMA(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))


# %%
# TODO: Inital difference with TA-Lib, maybe.
def test_t3():
    low, close, high = mock_data()
    vals = t3(close, 5, 0.7)
    ta_vals = ta.T3(close, 5, 0.7)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)])[-50:])


# %%
def test_kama():
    low, close, high = mock_data()
    vals = kama(close, 30)
    ta_vals = ta.KAMA(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))


# %%
def test_midprice():
    low, close, high = mock_data()
    vals = midprice(high, low, 30)
    ta_vals = ta.MIDPRICE(high, low, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))


# %%
def test_midpoint():
    low, close, high = mock_data()
    vals = midpoint(close, 30)
    ta_vals = ta.MIDPOINT(close, 30)
    assert np.all(np.isclose(vals[~np.isnan(vals)],
                             ta_vals[~np.isnan(ta_vals)]))


# %%
def test_bbands():
    low, close, high = mock_data()
    upb, mdb, lwb = bbands(close, 30)
    taupb, tamdb, talwb = ta.BBANDS(close, 30)
    assert np.all(np.isclose(upb[~np.isnan(upb)],
                             taupb[~np.isnan(taupb)]))
    assert np.all(np.isclose(mdb[~np.isnan(mdb)],
                             tamdb[~np.isnan(tamdb)]))
    assert np.all(np.isclose(lwb[~np.isnan(lwb)],
                             talwb[~np.isnan(talwb)]))


# %%
def test_sar():
    low, close, high = mock_data()
    vals = sar(high, low)
    ta_vals = ta.SAR(high, low)
    # assert np.all(np.isclose(vals[~np.isnan(vals)],
    #                          ta_vals[~np.isnan(ta_vals)]))


# %%
def test_sarext():
    low, close, high = mock_data()
    vals = sar_ext(high, low, 1, 0.1, 0.02, 0.02, 0.2, 0.01, 0.01, 0.2)
    ta_vals = ta.SAREXT(high, low, 1, 0.1, 0.02, 0.02, 0.2, 0.01, 0.01, 0.2)
    # assert np.all(np.isclose(vals[~np.isnan(vals)],
    #                          ta_vals[~np.isnan(ta_vals)]))


