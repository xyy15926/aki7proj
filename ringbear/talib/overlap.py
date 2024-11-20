#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: overlap.py
#   Author: xyy15926
#   Created: 2024-11-18 10:02:45
#   Updated: 2024-11-20 10:35:08
#   Description:
#   Ref: https://github.com/frgomes/ta-lib_code/blob/master/ta-lib/c/src/ta_func/
#   Ref: https://blog.csdn.net/weixin_43420026/article/details/118462233
# ---------------------------------------------------------

# %%
import numpy as np

def mock_data():
    dt = np.random.rand(100, 3)
    dt.sort(axis=1)
    return dt[:, 0], dt[:, 1], dt[:, 2]


# %%
low, close, high = mock_data()
timeperiod = 30
periods = np.random.randint(5, 20, len(close))


# %%
def ma(close, timeperiod=30, matype=0):
    """General moving average.

    Params:
    --------------------------
    timeperiod: Time period.
    matype: Moving average type.
      0: Simple MA.
      2: MA with weights of descending range distribution.
      3: MA with wieghts of triangular distribution.

    Return:
    --------------------------
    NDA of the `close` shape with `np.nan` filling the preceding non-defined.
    """
    # Triangular weights.
    triup = np.arange(1, (timeperiod + 1) // 2 + 1)
    tridn = np.arange((timeperiod + 1) // 2, 0, -1)
    tris = np.concatenate([triup, tridn])
    tris = tris / tris.sum()

    ws = {
        0: np.ones(timeperiod) / timeperiod,
        2: np.arange(timeperiod, 0, -1) / (timeperiod * (timeperiod + 1)) * 2,
        3: tris,
    }
    ret = np.ndarray(len(close))
    ret[:timeperiod - 1] = np.nan
    ret[timeperiod - 1:] = np.convolve(close, ws[matype], "valid")

    return ret


# %%
def sma(close, timeperiod=30):
    """Simple moving average.
    """
    ret = np.ndarray(len(close))
    ret[:timeperiod - 1] = np.nan
    ret[timeperiod - 1:] = np.convolve(close, np.ones(timeperiod), "valid") / timeperiod
    return ret


# %%
def wma(close, timeperiod=30):
    """Moving average with weights of descending range index, namely N - 1.
    """
    ws = np.arange(timeperiod, 0, -1)
    wss = timeperiod * (timeperiod + 1) / 2
    ret = np.convolve(close, ws, "valid") / wss
    ret = np.concatenate([[np.nan] * (timeperiod - 1), ret])

    return ret


# %%
def trima_yao(close, timeperiod=30):
    """Moving average with weights of triangular distribution.
    """
    return sma(sma(close, timeperiod // 2), timeperiod // 2 + 1)


def trima(close, timeperiod=30):
    """Moving average with weights of triangular distribution.
    """
    # Triangle weight.
    upop = np.arange(1, (timeperiod + 1) // 2 + 1)
    dnop = np.arange((timeperiod + 1) // 2, 0, -1)
    ops = np.concatenate([upop, dnop])
    ws = ops.sum()

    ret = np.ndarray(len(close))
    ret[:timeperiod - 1] = np.nan
    ret[timeperiod - 1:] = np.convolve(close, ops, "valid") / ws

    return ret


# %%
def mavp(close, periods, minperiods=2, maxperiods=30, matype=0):
    """Moving average with veriable periods.
    """
    ret = np.ndarray(len(close))
    MA = np.mean
    for i in range(len(close)):
        if i < maxperiods - 1:
            ret[i] = np.nan
            continue
        prd = max(minperiods, periods[i])
        prd = min(maxperiods, prd)
        ret[i] = MA(close[i - prd + 1: i + 1])
    return ret


# %%
def ema(close, timeperiod=30):
    """Exponential moving average.

    Pre-defined weights are not used here because of the precision of float.
    """
    # ws = np.power((1 - K), np.arange(timeperiod)) * ([K] * (timeperiod - 1) + [1])
    ret = np.ndarray(len(close))
    K = 2 / (timeperiod + 1)
    for i in range(len(close)):
        if i < timeperiod - 1:
            ret[i] = np.nan
            continue
        # Use SMA as the first element of EMA.
        elif i == timeperiod - 1:
            ret[i] = np.mean(close[:timeperiod])
        else:
            ret[i] = K * close[i] + (1 - K) * ret[i - 1]

    return ret


# %%
def dema(close, timeperiod=30):
    """Double exponential moving average.

    DEMA = 2 * EMA - EMA(EMA)
    """
    emav = ema(close, timeperiod)
    emavv = ema(emav[timeperiod - 1:], timeperiod)
    emav *= 2
    emav[timeperiod - 1:] -= emavv
    return emav


# %%
def tema(close, timeperiod=30):
    """Triple exponential moving average.

    TEMA = 3 * EMA - 3EMA(EMA) + EMA(EMA(EMA)))
    """
    emav = ema(close, timeperiod)
    emavv = ema(emav[timeperiod - 1:], timeperiod)
    emavvv = ema(emavv[timeperiod - 1:], timeperiod)
    emav *= 3
    emav[timeperiod - 1:] -= 3 * emavv
    emav[2 * timeperiod - 2:] += emavvv
    return emav


# %%
def gd(close, timeperiod=5, vfactor=0):
    """DEMA with variable coefficient of the `DOUBLE`.

    GD = (1 + VF) * EMA - VF * EMA(EMA)
    """
    emav = ema(close, timeperiod)
    emavv = ema(emav[timeperiod - 1:], timeperiod)
    emav *= (1 + vfactor)
    emav[timeperiod - 1:] -= emavv * vfactor
    return emav


def t3(close, timeperiod=5, vfactor=0):
    """Triple variable DEMA.

    T3 = GD(GD(GD)))
    """
    gdv = gd(close, timeperiod, vfactor)
    gdvv = gd(gdv[~np.isnan(gdv)], timeperiod, vfactor)
    gdvvv = gd(gdvv[~np.isnan(gdvv)], timeperiod, vfactor)
    ret = np.ndarray(len(close))
    ret[: 4 * timeperiod - 4] = np.nan
    ret[4 * timeperiod - 4:] = gdvvv
    return ret


# %%
def kama(close, timeperiod=30):
    """Kaufman's adaptive moving average.

    Adaptive coefficient of the EMA:
      1. diff_t = sum_{i=t-n+1}^t |close_i - close_{i-1}|
      2. er_t = |close_t - close_{t-n}| / diff_t
      3. alpha_t = (er_t * (afast - aslow) + alow) ** 2
      4. kama_t = alpha_t * close_t + (1 - alpha_t) * kama_{t-1}
    """
    # Coefficient of upper and lower threshold.
    afast = 2 / (2 + 1)
    aslow = 2 / (30 + 1)

    # Adaptive coefficients.
    diffs = np.abs(np.diff(close))
    ers = (np.abs(close[timeperiod:] - close[:-timeperiod])
           / np.convolve(diffs, np.ones(timeperiod), "valid"))
    alphas = (ers * (afast - aslow) + aslow) ** 2

    ret = np.ndarray(len(close))
    for i in range(len(close)):
        if i < timeperiod:
            ret[i] = np.nan
        # Use SMA as the first element of EMA.
        elif i == timeperiod:
            kn = i - timeperiod
            ret[i] = alphas[kn] * close[i] + (1 - alphas[kn]) * close[i - 1]
        else:
            kn = i - timeperiod
            ret[i] = alphas[kn] * close[i] + (1 - alphas[kn]) * ret[i - 1]

    return ret


# %%
def midprice(high, low, timeperiod=14):
    """Mean of the maximum of high and minimum of low.
    """
    ret = np.ndarray(len(high))
    for i in range(len(high)):
        if i < timeperiod - 1:
            ret[i] = np.nan
        else:
            ret[i] = (np.max(high[i - timeperiod + 1: i + 1])
                      + np.min(low[i - timeperiod + 1: i + 1])) / 2
    return ret


# %%
def midpoint(close, timeperiod=14):
    """Mean of the maximum and minimum.
    """
    ret = np.ndarray(len(close))
    for i in range(len(close)):
        if i < timeperiod - 1:
            ret[i] = np.nan
        else:
            ret[i] = (np.max(close[i - timeperiod + 1: i + 1])
                      + np.min(close[i - timeperiod + 1: i + 1])) / 2
    return ret


# %%
def bbands(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
    """Moving average and upper and lower bound adjusted by stdvar.
    """
    mid = sma(close, timeperiod)
    stds = np.ndarray(len(close))
    for i in range(len(close)):
        if i < timeperiod - 1:
            stds[i] = np.nan
        else:
            stds[i] = np.std(close[i - timeperiod + 1: i + 1])
    upperband = mid + nbdevup * stds
    lowerband = mid - nbdevdn * stds

    return upperband, mid, lowerband


# %%
def sar(high, low, acceleration=0.02, maximum=0.2):
    """ Stop and reverse.

    1. Accelarating trend if thresholds are exceeds.
    2. Or keep the trend unchanged.
    3. Reverse when sar cross over with high or low.
    """
    # 1: long trend, -1: short trend
    trend = 1
    ret = np.ndarray(len(high))
    ret[0] = np.nan
    sar = low[0] if trend == 1 else high[0]
    ep = high[1] if trend == 1 else low[1]
    af = acceleration

    for i in range(1, len(high)):
        if trend == 1:
            # Change trend from LONG to SHORT.
            if sar >= low[i]:
                sar = max(ep, high[i], high[i - 1])
                ret[i] = sar
                trend = -1
                af = acceleration
                ep = low[i]
                sar = max(sar + af * (ep - sar), high[i], high[i - 1])
            else:
                ret[i] = sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + acceleration, maximum)
                sar = min(sar + af * (ep - sar), low[i], low[i - 1])
        else:
            # Change trend from SHORT to LONG.
            if sar <= high[i]:
                sar = min(ep, low[i], low[i - 1])
                ret[i] = sar
                trend = 1
                af = acceleration
                ep = high[i]
                sar = min(sar + af * (ep - sar), low[i], low[i - 1])
            else:
                ret[i] = sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + acceleration, maximum)
                sar = max(sar + af * (ep - sar), high[i], high[i - 1])

    return ret


# %%
def sar_ext(high, low, startvalue=1,
            offsetonreverse=0,
            accelerationinitlong=0,
            accelerationlong=0,
            accelerationmaxlong=0,
            accelerationinitshort=0,
            accelerationshort=0,
            accelerationmaxshort=0):
    """ Stop and reverse.

    1. Accelarating trend if thresholds are exceeds.
    2. Or keep the trend unchanged.
    3. Reverse when sar cross over with high or low.
    """
    # 1: long trend, -1: short trend
    trend = startvalue
    ret = np.ndarray(len(high))
    ret[0] = np.nan
    if trend == 1:
        sar = low[0]
        ep = high[1]
        af = accelerationinitlong
    else:
        sar = high[0]
        ep = low[1]
        af = accelerationinitshort

    for i in range(1, len(high)):
        if trend == 1:
            # Change trend from LONG to SHORT.
            if sar >= low[i]:
                sar = max(ep, high[i], high[i - 1]) * (1 + offsetonreverse)
                ret[i] = -sar
                trend = -1
                af = accelerationinitshort
                ep = low[i]
                sar = max(sar + af * (ep - sar), high[i], high[i - 1])
            else:
                ret[i] = sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + accelerationlong, accelerationmaxlong)
                sar = min(sar + af * (ep - sar), low[i], low[i - 1])
        else:
            # Change trend from SHORT to LONG.
            if sar <= high[i]:
                sar = min(ep, low[i], low[i - 1]) * (1 + offsetonreverse)
                ret[i] = sar
                trend = 1
                af = accelerationinitlong
                ep = high[i]
                sar = min(sar + af * (ep - sar), low[i], low[i - 1])
            else:
                ret[i] = -sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + accelerationshort, accelerationmaxshort)
                sar = max(sar + af * (ep - sar), high[i], high[i - 1])

    return ret


# %%
def ht_trendline(close):
    pass


# %%
def mama(close):
    pass
