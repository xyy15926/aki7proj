#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: momentum.py
#   Author: xyy15926
#   Created: 2024-11-22 08:28:32
#   Updated: 2024-11-25 13:54:59
#   Description:
#   Ref: https://github.com/frgomes/ta-lib_code/blob/master/ta-lib/c/src/ta_func/
#   Ref: https://www.fmlabs.com/reference/default.htm
#   Ref: https://blog.csdn.net/weixin_43420026/article/details/118637030
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, Any

import logging
import numpy as np

from ringbear.talib.overlap import (
    ema, ma, sma
)

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def macd(
    close: np.ndarray,
    fastperiod:int = 12,
    slowperiod:int = 26,
    signalperiod:int = 9,
) -> tuple[np.ndarray]:
    """Exponential moving average convergence and divergence.

    Compare the divergence between the fast EMA with shorter period and
    the slow EMA with longer period.
    1. The calculation of the EMA must start at the same point, namely the
      first simple mean must be start at the `slowperiod - 1`, though the
      preceding values will be "dropped" for fast EMA.
    2. MACD is the diff between the fast and slow EMA.
    3. Signal is the EMA of the MACD, and the MACD's preceding values will be
      set with `np.nan` to keep consistent.

    Params:
    -------------------------
    close: Close price.
    fastperiod: Time period for fast EMA.
    slowperiod: Time period for slow EMA.
    signalperiod: Time period for signal.

    Return:
    -------------------------
    MACD: np.ndarray with `np.nan` as the preceding `slowperiod + signalperiod - 2`
      items.
    MACD signal: Ditto.
    MACD histagram: Ditto.
    """
    # Fast MV.
    fastma = np.ndarray(len(close))
    fast_start = slowperiod - fastperiod
    fastma[:fast_start] = np.nan
    fastma[fast_start:] = ema(close[fast_start:], fastperiod)

    # Slow MV.
    slowma = ema(close, slowperiod)

    # MV divergence.
    macd = fastma - slowma

    # Signal and hist.
    macdsignal = np.ndarray(len(close))
    macdsignal[:slowperiod - 1] = np.nan
    macdsignal[slowperiod - 1:] = ema(macd[slowperiod - 1:], signalperiod)
    macdhist = macd - macdsignal

    # Set `np.nan` to keep the pace with MACDSignal.
    macd[:slowperiod + signalperiod - 2] = np.nan

    return macd, macdsignal, macdhist


# %%
def macd_ext(
    close: np.ndarray,
    fastperiod:int = 12,
    fastmatype: int = 0,
    slowperiod:int = 26,
    slowmatype: int = 0,
    signalperiod:int = 9,
    signalmatype: int = 0,
) -> tuple[np.ndarray]:
    """Moving average convergence and divergence.

    Just like MACD, but the MA type could be speicfied instead of the predefed
    EMA.

    Compare the divergence between the fast MA with shorter period and
    the slow MA with longer period.
    1. The calculation of the MA must start at the same point, namely the
      first simple mean must be start at the `slowperiod - 1`, though the
      preceding values will be "dropped" for fast MA.
    2. MACD is the diff between the fast and slow MA.
    3. Signal is the MA of the MACD, and the MACD's preceding values will be
      set with `np.nan` to keep consistent.

    Params:
    -------------------------
    close: Close price.
    fastperiod: Time period for fast MA.
    slowperiod: Time period for slow MA.
    signalperiod: Time period for signal.

    Return:
    -------------------------
    MACD: np.ndarray with `np.nan` as the preceding `slowperiod + signalperiod - 2`
      items.
    MACD signal: Ditto.
    MACD histagram: Ditto.
    """
    # Fast MV.
    fastma = np.ndarray(len(close))
    fast_start = slowperiod - fastperiod
    fastma[:fast_start] = np.nan
    fastma[fast_start:] = ma(close[fast_start:], fastperiod, fastmatype)

    # Slow MV.
    slowma = ma(close, slowperiod, slowmatype)

    # MV divergence.
    macd = fastma - slowma

    # Signal and hist.
    macdsignal = np.ndarray(len(close))
    macdsignal[:slowperiod - 1] = np.nan
    macdsignal[slowperiod - 1:] = ma(macd[slowperiod - 1:],
                                     signalperiod,
                                     signalmatype)
    macdhist = macd - macdsignal

    # Set `np.nan` to keep the pace with MACDSignal.
    macd[:slowperiod + signalperiod - 2] = np.nan

    return macd, macdsignal, macdhist


# %%
def cci(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timeperiod: int = 14
) -> np.ndarray:
    """Commodity Channel Index.

    Ref: <https://www.litefinance.org/zh/blog/for-beginners/zui-jia-wai-hui-zhi-biao/shang-pin-tong-dao-zhi-shu-cci/>

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    CCI: np.ndarray with preceding `timeperiod - 1` np.nan
    """
    tp = (high + low + close) / 3

    # Calculate SMA and rolling STD.
    smav = sma(tp, timeperiod)
    mads = np.ndarray(len(close))
    # ATTENTION: MAD instead of STD.
    mads[:timeperiod - 1] = np.nan
    for i in range(timeperiod - 1, len(close)):
        mads[i] = (np.abs(tp[i - timeperiod + 1: i + 1] - smav[i]).sum()
                   / timeperiod)

    cci = (tp - smav) / mads / 0.015

    return cci


# %%
def trix(
    close: np.ndarray,
    timeperiod: int = 30
) -> np.ndarray:
    """Triple Exponentially Smoothed Average.

    Params:
    --------------------------
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    TRIX: np.ndarray with preceding  `3 * timeperiod - 2` np.nan
    """
    emav = ema(close, timeperiod)
    emavv = ema(emav[timeperiod - 1:], timeperiod)
    emavvv = ema(emavv[timeperiod - 1:], timeperiod)
    trix = np.ndarray(len(close))
    trix[:2 * timeperiod - 2 + 1] = np.nan
    trix[2 * timeperiod - 2 + 1:] = (emavvv[1:] - emavvv[:-1]) / emavvv[:-1] * 100

    return trix


# %%
def bop(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Balance of power.

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    BOP: np.ndarray with no np.nan.
    """
    return (close - open_) / (high - low)


# %%
def rsi(
    close: np.ndarray,
    timeperiod: int = 14
) -> np.ndarray:
    """Relative Strength Index.

    Params:
    --------------------------
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    RSI: np.ndarray with preceding `timeperiod` np.nan.
    """
    diffs = close[1:] - close[:-1]
    ups = np.where(diffs > 0, diffs, 0)
    dns = -np.where(diffs < 0, diffs, 0)

    # NOTE: The EMA here are not called with defualt exponential coefficient.
    upmav = ema(ups, timeperiod, K=1 / timeperiod)
    dnmav = ema(dns, timeperiod, K=1 / timeperiod)

    rsiv = np.ndarray(len(close))
    rsiv[0] = np.nan
    rsiv[1:] = upmav / (dnmav + upmav) * 100

    return rsiv


# %%
def cmo(
    close: np.ndarray,
    timeperiod: int = 14
) -> np.ndarray:
    """Chande Momentum Oscillator.

    CMO is a modified RSI with deviding total movement with net movment.
    NOTE: The 

    Params:
    --------------------------
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    CMO: np.ndarray with preceding `timeperiod` np.nan.
    """
    diffs = close[1:] - close[:-1]
    ups = np.where(diffs > 0, diffs, 0)
    dns = -np.where(diffs < 0, diffs, 0)

    # The formula in FMLabs for CMO is SMA, but the TA-Lib use modified EMA,
    # just the same as the RSI.
    upmav = ema(ups, timeperiod, K=1 / timeperiod)
    dnmav = ema(dns, timeperiod, K=1 / timeperiod)

    cmov = np.ndarray(len(close))
    cmov[0] = np.nan
    # Diff with RSI: Net movement.
    cmov[1:] = (upmav - dnmav) / (dnmav + upmav) * 100

    return cmov


# %%
def ultosc(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> np.ndarray:
    """Ultimate Oscillator.

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    ULTOSC: np.ndarray with preceding `timeperiod - 1` np.nan.
    """
    true_low = np.minimum(low[1:], close[:-1])
    true_range = np.maximum(high[1:], close[:-1]) - true_low
    a1 = sma(close[1:] - true_low, timeperiod1) * timeperiod1
    a2 = sma(close[1:] - true_low, timeperiod2) * timeperiod2
    a3 = sma(close[1:] - true_low, timeperiod3) * timeperiod3
    b1 = sma(true_range, timeperiod1) * timeperiod1
    b2 = sma(true_range, timeperiod2) * timeperiod2
    b3 = sma(true_range, timeperiod3) * timeperiod3

    ult_osc = np.ndarray(len(high))
    ult_osc[0] = np.nan
    # Reallocate weigths of SMA with different period.
    ult_osc[1:] = ((a1 / b1 * 4) + (a2 / b2) * 2 + a3 / b3) / 7 * 100

    return ult_osc


# %%
def mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """Money Flow Index.

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    volume: Money flow volume.
    timeperiod: Time period.

    Return:
    --------------------------
    MFI: np.ndarray with preceding `timeperiod` np.nan.
    """
    typical_price = (high + low + close) / 3
    tp_diff = typical_price[1:] - typical_price[:-1]
    mf = typical_price * volume
    pos_mf = np.where(tp_diff > 0, mf[1:], 0)
    neg_mf = np.where(tp_diff < 0, mf[1:], 0)
    pos_mf_ma = sma(pos_mf, timeperiod)
    neg_mf_ma = sma(neg_mf, timeperiod)

    mfiv = np.ndarray(len(close))
    mfiv[0] = np.nan
    mfiv[1:] = pos_mf_ma / (pos_mf_ma + neg_mf_ma) * 100

    return mfiv


# %%
def aroon(
    high: np.ndarray,
    low: np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """AROON Index.

    Params:
    --------------------------
    high: High price.
    low: Low price.
    timeperiod: Time period.

    Return:
    --------------------------
    AROON DOWN: np.ndarray with preceding `timeperiod` np.nan.
    AROON UP: np.ndarray with preceding `timeperiod` np.nan.
    """
    aroon_up = np.ndarray(len(high))
    aroon_up[:timeperiod] = np.nan
    aroon_dn = np.ndarray(len(high))
    aroon_dn[:timeperiod] = np.nan

    # TODO: More efficient way.
    for i in range(timeperiod, len(high)):
        # Current price is included, namely `timeperiod + 1` items are
        # compared to get the maximum or minimum.
        hpos = np.argmax(high[i - timeperiod: i + 1])
        aroon_up[i] = hpos / timeperiod * 100
        lpos = np.argmin(low[i - timeperiod: i + 1])
        aroon_dn[i] = lpos / timeperiod * 100

    return aroon_dn, aroon_up


# %%
def aroon_osc(
    high: np.ndarray,
    low: np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """AROON Oscillator.

    Params:
    --------------------------
    high: High price.
    low: Low price.
    timeperiod: Time period.

    Return:
    --------------------------
    AROON OSC: np.ndarray with preceding `timeperiod` np.nan.
    """
    adn, aup = aroon(high, low, timeperiod)
    return aup - adn


# %%
def dm(
    high: np.ndarray,
    low: np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """Directional Movement.

    Ref: https://github.com/frgomes/ta-lib_code/blob/master/ta-lib/c/src/ta_func/ta_PLUS_DM.c

    Params:
    --------------------------
    high: High price.
    low: Low price.
    timeperiod: Time period.

    Return:
    --------------------------
    PLUS_DM: np.ndarray with preceding `timeperiod` np.nan.
    MIUNS_DM: np.ndarray with preceding `timeperiod` np.nan.
    """
    dh = high[1:] - high[:-1]
    dl = low[:-1] - low[1:]

    # PLUS_DM
    pos_dm = np.ndarray(len(high))
    pos_dm[0] = 0
    pos_dm[1:] = np.where((dh > dl) & (dh > 0), dh, 0)
    pos_dmma = ema(pos_dm, timeperiod, K=1 / timeperiod) * timeperiod

    # MINUS_DM
    neg_dm = np.ndarray(len(high))
    neg_dm[0] = 0
    neg_dm[1:] = np.where((dh < dl) & (dl > 0), dl, 0)
    neg_dmma = ema(neg_dm, timeperiod, K=1 / timeperiod) * timeperiod

    return pos_dmma, neg_dmma


# %%
def di(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndrray,
    timeperiod: int = 14,
) -> np.ndarray:
    """Directional Movement Index.

    Ref: https://github.com/frgomes/ta-lib_code/blob/master/ta-lib/c/src/ta_func/ta_PLUS_DI.c
    The TR14's definition in the comment is different from the final
    implementation. And the first TR13 is dropped forcedly.

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    PLUS_DM: np.ndarray with preceding `timeperiod` np.nan.
    MIUNS_DM: np.ndarray with preceding `timeperiod` np.nan.
    """
    pos_dmma, neg_dmma = dm(high, low, timeperiod)

    # True range
    true_high = np.maximum(high[1:], close[:-1])
    true_low = np.minimum(low[1:], close[:-1])
    true_range = np.ndarray(len(high))
    # `TRange[0] = 0` is set here to reuse the EMA, or the first element of
    # TRange MA should be sum the preceding `timeperiod - 1` TRanges.
    true_range[0] = 0
    true_range[1:] = true_high - true_low
    trma = ema(true_range, timeperiod, K=1 / timeperiod) * timeperiod
    # Set `np.nan` to keep alignment.
    trma[timeperiod - 1] = np.nan

    pdi = pos_dmma / trma * 100
    ndi = neg_dmma / trma * 100

    return pdi, ndi


# %%
def dx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndrray,
    timeperiod: int = 14,
) -> np.ndarray:
    """Directional Movement Index.

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    DX: np.ndarray with preceding `timeperiod` np.nan.
    """
    pdi, ndi = di(high, low, close, timeperiod)
    dxv = np.abs(pdi - ndi) / (pdi + ndi) * 100

    return dxv


# %%
def adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndrray,
    timeperiod: int = 14,
) -> np.ndarray:
    """Average Directional Movement Index.

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    ADX: np.ndarray with preceding `2 * timeperiod - 1` np.nan.
    """
    dxv = dx(high, low, close, timeperiod)
    adxv = np.ndarray(len(high))
    adxv[:timeperiod] = np.nan
    adxv[timeperiod:] = ema(dxv[timeperiod:], timeperiod, K=1 / timeperiod)

    return adxv


# %%
def adxr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndrray,
    timeperiod: int = 14,
) -> np.ndarray:
    """Average Directional Movement Rating.

    Formula:
    ADXR_n = (ADX_n + ADX_{n - timeperiod + 1}) / 2
    The formula in FMLib:
    ADXR_n = (ADX_n + ADX_{n - timeperiod}) / 2

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    ADX: np.ndarray with preceding `3 * timeperiod - 2` np.nan.
    """
    adxv = adx(high, low, close, timeperiod)
    adxrv = np.ndarray(len(close))
    adxrv[:3 * timeperiod - 2] = np.nan
    adxrv[3 * timeperiod - 2:] = (adxv[3 * timeperiod - 2:]
                                  + adxv[2 * timeperiod - 1: -timeperiod + 1]) / 2

    return adxrv


# %%
def mom(
    close: np.ndarray,
    timeperiod: int = 10
) -> np.ndarray:
    """Momentum.

    Params:
    --------------------------
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    MOM: np.ndarray with preceding `timeperiod` np.nan.
    """
    ret = np.ndarray(len(close))
    ret[:timeperiod] = np.nan
    ret[timeperiod:] = close[timeperiod:] - close[:-timeperiod]
    return ret


# %%
def willr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """Williams %R.

    Formula here and in FMLib:
    WILLR = (max(high[:i]) - close[i]) / (max(high[:i]) - min(low[:i]))
    Formula in TA-lib:
    WILLR = (max(high[:i]) - close[i]) / -(max(high[:i]) - min(low[:i]))

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    WILLR: np.ndarray with preceding `timeperiod - 1` np.nan.
    """
    ret = np.ndarray(len(close))
    ret[:timeperiod - 1] = np.nan
    for i in range(timeperiod - 1, len(close)):
        hhigh = high[i - timeperiod + 1: i + 1].max()
        llow = low[i - timeperiod + 1: i + 1].min()
        ret[i] = (hhigh - close[i]) / (hhigh - llow) * 100

    return ret


# %%
def stochf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> np.ndarray:
    """Stochastic Oscillator with Fast K and D.

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    fastk_period:
    fastd_period:
    fastd_matype:

    Return:
    --------------------------
    FastK: np.ndarray with preceding `fastk_period + fastd_period - 2` np.nan.
    FastD: np.ndarray with preceding `fastk_period + fastd_period - 2` np.nan.
    """
    # Raw K or Fast K or Unsmoothed K
    raw_k = np.ndarray(len(close))
    raw_k[:fastk_period - 1] = np.nan
    for i in range(fastk_period - 1, len(close)):
        hhigh = high[i - fastk_period + 1: i + 1].max()
        llow = low[i - fastk_period + 1: i + 1].min()
        raw_k[i] = (close[i] - llow) / (hhigh - llow) * 100

    # Fast D or smoothed K or Slow K(with more np.nan).
    fast_d = np.ndarray(len(close))
    fast_d[fastk_period - 1:] = ma(raw_k[fastk_period - 1:],
                                   fastd_period, fastd_matype)
    fast_d[:fastk_period - 1] = np.nan

    # Set the preceding items in FastK, SlowK with np.nan to keep alignment
    # with the FastD, SlowD.
    # Code here just to emphasize this.
    raw_k[:fastk_period + fastd_period - 2] = np.nan

    return raw_k, fast_d


# %%
def stoch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_matype: int = 0,
    slowd_period: int = 3,
    slowd_matype: int = 0
) -> np.ndarray:
    """Stochastic Oscillator with Slow K and D.

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    fastk_period:
    slowk_period:
    slowk_matype:
    slowd_period:
    slowd_matype:

    Return:
    --------------------------
    SlowK: np.ndarray with preceding `fastk_period + slowk_period + slowd_period - 3` np.nan.
    SlowD: np.ndarray with preceding `fastk_period + slowk_period + slowd_period - 3` np.nan.
    """
    fast_k, fast_d = stochf(high, low, close,
                            fastk_period,
                            slowk_period, slowk_matype)

    # Slow D or smoothed Slow K
    slow_d = np.ndarray(len(close))
    slow_d[fastk_period + slowk_period - 2:] = (
        ma(fast_d[fastk_period + slowk_period - 2:],
           slowd_period, slowd_matype))
    slow_d[:fastk_period + slowk_period - 2] = np.nan

    # Set the preceding items in FastK, SlowK with np.nan to keep alignment
    # with the FastD, SlowD.
    # Code here just to emphasize this.
    slow_k = fast_d.copy()
    slow_k[:fastk_period + slowk_period + slowd_period - 3] = np.nan

    return slow_k, slow_d


# %%
def stoch_rsi(
    close: np.ndarray,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0
) -> np.ndarray:
    """Stochastic RSI.

    Params:
    --------------------------
    high: High price.
    low: Low price.
    close: Close price.
    timeperiod: Time period for RSI.
    fastk_period:
    fastd_period:
    fastd_matype:

    Return:
    --------------------------
    FastK: np.ndarray with preceding `fastk_period + fastd_period - 2` np.nan.
    FastD: np.ndarray with preceding `fastk_period + fastd_period - 2` np.nan.
    """
    rsiv = rsi(close, timeperiod)
    fast_k = np.ndarray(len(close))
    fast_d = np.ndarray(len(close))
    fast_k[:timeperiod] = np.nan
    fast_d[:timeperiod] = np.nan
    fast_k[timeperiod:], fast_d[timeperiod:] = stochf(rsiv[timeperiod:],
                                                      rsiv[timeperiod:],
                                                      rsiv[timeperiod:],
                                                      fastk_period,
                                                      fastd_period,
                                                      fastd_matype)
    return fast_k, fast_d
