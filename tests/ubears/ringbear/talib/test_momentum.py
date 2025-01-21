#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_momentum.py
#   Author: xyy15926
#   Created: 2024-11-22 09:53:05
#   Updated: 2025-01-21 15:06:45
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
    from ubears.ringbear.talib import momentum
    reload(overlap)
    reload(momentum)

from ubears.ringbear.talib.momentum import (
    macd, macd_ext, cci, trix,
    bop, rsi, ultosc, mfi, cmo,
    aroon, aroon_osc,
    dm, di, dx, adx, adxr,
    mom, willr,
    stochf, stoch, stoch_rsi,
)


# %%
def mock_data():
    dt = np.random.rand(100, 3)
    dt.sort(axis=1)
    return dt[:, 2], dt[:, 1], dt[:, 0]


def mock_data_all():
    high, close, low = mock_data()
    high_, open_, low_ = mock_data()
    high = np.where(high > high_, high, high_)
    low = np.where(low < low_, low, low_)
    volume = np.random.rand(100) * 10000

    return open_, high, low, close, volume


# %%
def test_macd():
    high, close, low = mock_data()
    vmacd, vmacdsignal, vmacdhist = macd(close, 12, 26, 9)
    ta_macd, ta_macdsignal, ta_macdhist = ta.MACD(close, 12, 26, 9)

    assert np.all(np.isclose(vmacd[~np.isnan(vmacd)],
                             ta_macd[~np.isnan(ta_macd)]))
    assert np.all(np.isclose(vmacdsignal[~np.isnan(vmacdsignal)],
                             ta_macdsignal[~np.isnan(ta_macdsignal)]))
    assert np.all(np.isclose(vmacdhist[~np.isnan(vmacdhist)],
                             ta_macdhist[~np.isnan(ta_macdhist)]))


# %%
def test_macd_ext():
    high, close, low = mock_data()
    vmacd, vmacdsignal, vmacdhist = macd_ext(close, 12, 0, 26, 0, 9, 2)
    ta_macd, ta_macdsignal, ta_macdhist = ta.MACDEXT(close, 12, 0, 26, 0, 9, 2)

    assert np.all(np.isclose(vmacd[~np.isnan(vmacd)],
                             ta_macd[~np.isnan(ta_macd)]))
    assert np.all(np.isclose(vmacdsignal[~np.isnan(vmacdsignal)],
                             ta_macdsignal[~np.isnan(ta_macdsignal)]))
    assert np.all(np.isclose(vmacdhist[~np.isnan(vmacdhist)],
                             ta_macdhist[~np.isnan(ta_macdhist)]))


# %%
def test_cci():
    high, close, low = mock_data()
    val = cci(high, low, close, 30)
    ta_val = ta.CCI(high, low, close, 30)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_trix():
    high, close, low = mock_data()
    val = trix(close, 14)
    ta_val = ta.TRIX(close, 14)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_bop():
    open_, high, low, close, volume = mock_data_all()
    val = bop(open_, high, low, close)
    ta_val = ta.BOP(open_, high, low, close)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_rsi():
    high, close, low = mock_data()
    val = rsi(close, 30)
    ta_val = ta.RSI(close, 30)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_cmo():
    high, close, low = mock_data()
    val = cmo(close, 30)
    ta_val = ta.CMO(close, 30)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_ultosc():
    high, close, low = mock_data()
    val = ultosc(high, low, close, 7, 14, 28)
    ta_val = ta.ULTOSC(high, low, close, 7, 14, 28)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_mfi():
    open_, high, low, close, volume = mock_data_all()
    val = mfi(high, low, close, volume, 14)
    ta_val = ta.MFI(high, low, close, volume, 14)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_aroon():
    open_, high, low, close, volume = mock_data_all()
    adn, aup = aroon(high, low, 14)
    ta_adn, ta_aup = ta.AROON(high, low, 14)

    assert np.all(np.isclose(adn, ta_adn, equal_nan=True))
    assert np.all(np.isclose(aup, ta_aup, equal_nan=True))


# %%
def test_aroon_osc():
    open_, high, low, close, volume = mock_data_all()
    val = aroon_osc(high, low, 14)
    ta_val = ta.AROONOSC(high, low, 14)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_dm():
    open_, high, low, close, volume = mock_data_all()
    pos_dm, neg_dm = dm(high, low, 14)
    ta_pos = ta.PLUS_DM(high, low, 14)
    ta_neg = ta.MINUS_DM(high, low, 14)

    assert np.all(np.isclose(pos_dm, ta_pos, equal_nan=True))
    assert np.all(np.isclose(neg_dm, ta_neg, equal_nan=True))


# %%
def test_di():
    open_, high, low, close, volume = mock_data_all()
    pos_di, neg_di = di(high, low, close, 14)
    ta_pos = ta.PLUS_DI(high, low, close, 14)
    ta_neg = ta.MINUS_DI(high, low, close, 14)

    assert np.all(np.isclose(pos_di, ta_pos, equal_nan=True))
    assert np.all(np.isclose(neg_di, ta_neg, equal_nan=True))


# %%
def test_dx():
    open_, high, low, close, volume = mock_data_all()
    val = dx(high, low, close, 14)
    ta_val = ta.DX(high, low, close, 14)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_adx():
    open_, high, low, close, volume = mock_data_all()
    val = adx(high, low, close, 14)
    ta_val = ta.ADX(high, low, close, 14)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_adxr():
    open_, high, low, close, volume = mock_data_all()
    val = adxr(high, low, close, 14)
    ta_val = ta.ADXR(high, low, close, 14)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_mom():
    open_, high, low, close, volume = mock_data_all()
    val = mom(close, 10)
    ta_val = ta.MOM(close, 10)

    assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_willr():
    open_, high, low, close, volume = mock_data_all()
    val = willr(high, low, close, 14)
    ta_val = ta.WILLR(high, low, close, 14)

    assert np.all(np.isclose(val, -ta_val, equal_nan=True))


# %%
def test_stochf():
    open_, high, low, close, volume = mock_data_all()
    fast_k, fast_d = stochf(high, low, close, 5, 10, 0)
    tafk, tafd = ta.STOCHF(high, low, close, 5, 10, 0)

    assert np.all(np.isclose(fast_k, tafk, equal_nan=True))
    assert np.all(np.isclose(fast_d, tafd, equal_nan=True))


# %%
def test_stoch():
    open_, high, low, close, volume = mock_data_all()
    slow_k, slow_d = stoch(high, low, close, 5, 10, 0, 14, 0)
    task, tasd = ta.STOCH(high, low, close, 5, 10, 0, 14, 0)

    assert np.all(np.isclose(slow_k, task, equal_nan=True))
    assert np.all(np.isclose(slow_d, tasd, equal_nan=True))


# %%
def test_stoch_rsi():
    open_, high, low, close, volume = mock_data_all()
    fast_k, fast_d = stoch_rsi(close, 14, 5, 10, 0)
    tafk, tafd = ta.STOCHRSI(close, 14, 5, 10, 0)

    assert np.all(np.isclose(fast_k, tafk, equal_nan=True))
    assert np.all(np.isclose(fast_d, tafd, equal_nan=True))
