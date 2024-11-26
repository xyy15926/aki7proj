#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_candlestick.py
#   Author: xyy15926
#   Created: 2024-11-25 13:59:19
#   Updated: 2024-11-26 15:23:01
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import talib as ta
import numpy as np
if __name__ == "__main__":
    from importlib import reload
    from ringbear.talib import overlap
    from ringbear.talib import momentum
    from ringbear.talib import candlestick
    reload(overlap)
    reload(momentum)
    reload(candlestick)

from ringbear.talib.candlestick import (
    sma_excur,
    belthold, closing_marubozu, doji, dragonfly_doji, gravestone_doji,
    counter_attack, darkcloud_cover, doji_star, engulfing,
    hammer, hangingman,
)


# %%
def mock_data(n: int = 100):
    dt = np.random.rand(n, 3)
    dt.sort(axis=1)
    return dt[:, 2], dt[:, 1], dt[:, 0]


def mock_data_all(n: int = 100):
    high, close, low = mock_data(n)
    high_, open_, low_ = mock_data(n)
    high = np.where(high > high_, high, high_)
    low = np.where(low < low_, low, low_)
    volume = np.random.rand(n) * 10000

    return open_, high, low, close, volume


TEST_N = 1000


# %%
def test_belthold():
    for i in range(TEST_N):
        open_, high, low, close, volume = mock_data_all()
        val = belthold(open_, high, low, close)
        ta_val = ta.CDLBELTHOLD(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_closing_marubozu():
    for i in range(TEST_N):
        open_, high, low, close, volume = mock_data_all()
        val = closing_marubozu(open_, high, low, close)
        ta_val = ta.CDLCLOSINGMARUBOZU(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_doji():
    for i in range(TEST_N):
        open_, high, low, close, volume = mock_data_all()
        val = doji(open_, high, low, close)
        ta_val = ta.CDLDOJI(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_dragonfly_doji():
    for i in range(TEST_N):
        open_, high, low, close, volume = mock_data_all()
        val = dragonfly_doji(open_, high, low, close)
        ta_val = ta.CDLDRAGONFLYDOJI(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_gravestone_doji():
    for i in range(TEST_N):
        open_, high, low, close, volume = mock_data_all()
        val = gravestone_doji(open_, high, low, close)
        ta_val = ta.CDLGRAVESTONEDOJI(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_counter_attack():
    for i in range(TEST_N):
        open_, high, low, close, volume = mock_data_all()
        val = counter_attack(open_, high, low, close)
        ta_val = ta.CDLCOUNTERATTACK(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_darkcloud_cover():
    for i in range(TEST_N):
        open_, high, low, close, volume = mock_data_all()
        val = darkcloud_cover(open_, high, low, close, 0.5)
        ta_val = ta.CDLDARKCLOUDCOVER(open_, high, low, close, 0.5)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_doji_star():
    for i in range(TEST_N):
        open_, high, low, close, volume = mock_data_all()
        val = doji_star(open_, high, low, close)
        ta_val = ta.CDLDOJISTAR(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_engulfing():
    for i in range(TEST_N):
        open_, high, low, close, volume = mock_data_all()
        val = engulfing(open_, high, low, close)
        ta_val = ta.CDLENGULFING(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_hammer():
    for i in range(TEST_N):
        open_, high, low, close, volume = mock_data_all()
        val = hammer(open_, high, low, close)
        ta_val = ta.CDLHAMMER(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))


# %%
def test_hangingman():
    for i in range(TEST_N):
        open_, high, low, close, volume = mock_data_all()
        val = hangingman(open_, high, low, close)
        ta_val = ta.CDLHANGINGMAN(open_, high, low, close)

        assert np.all(np.isclose(val, ta_val, equal_nan=True))
