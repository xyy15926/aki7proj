#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: candlestick.py
#   Author: xyy15926
#   Created: 2024-11-25 13:44:52
#   Updated: 2024-11-29 12:09:37
#   Description:
#   Ref: https://github.com/frgomes/ta-lib_code/blob/master/ta-lib/c/src/ta_func/
#   Ref: https://www.fmlabs.com/reference/default.htm
#   Ref: https:https://blog.csdn.net/weixin_43420026/article/details/126743440
#   Ref: https://blog.csdn.net/suiyingy/article/details/118661718
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
def sma_excur(
    close: np.ndarray,
    timeperiod: 10,
) -> np.ndarray:
    """Simple moving average without current item.

    Just like shift 1 forward on the original SMA.

    Params:
    --------------------------
    close: Close price.
    timeperiod: Time period.

    Return:
    --------------------------
    SMA: np.ndarray with preceding `timeperiod` np.nan.
    """
    ret = np.ndarray(len(close))
    ret[:timeperiod] = np.nan
    ret[timeperiod:] = (np.convolve(close, np.ones(timeperiod), "valid")[:-1]
                        / timeperiod)
    return ret


# %%
def candle_spec(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    spec: str,
) -> np.ndarray:
    """Candle stick basic specifications.

    Global Setting: https://github.com/frgomes/ta-lib_code/blob/master/ta-lib/c/src/ta_common/ta_global.c#L128

    1 Candle Terms:
    ---------------------------
    - white: close > open_
    - black: close < open_
    - real body: abs(close - open_)
    - upper shadow: high - max(close, open_)
    - lower shadow: min(close, open_) - low

    2 Candle Terms:
    ---------------------------
    - declining: close[t] < close[t-1]
    - rising: close[t] > close[t-1]
    - gap: range gap
      - gap up: low[t] > high[t-1]
      - gap dn: high[t] < low[t-1]
    - star: body gapping up in a uptrend or down in a downtrend
      - gap-up: (close[t-1] > open_[t-1])
                & (min(close[t], open_[t]) > close[t-1])
      - gap-dn: (close[t-1] < open_[t-1])
                & (max(close[t], open_[t]) < close[t-1])
    - body engulf
      - white engulf black: (close[:-1] < open_[:-1])
                            & (close[:-1] > open_[1:])
                            & (open_[:-1] < close[1:])
      - black engulf white: (open_[:-1] < close[:-1])
                            & (close[:-1] < open_[1:])
                            & (open_[:-1] > close[1:])

    Possible description:
    ---------------------------
    - bodylong
    - bodyverylong
    - bodyshort
    - bodydoji
    - shadowlong
    - shadowverylong
    - shadowshort
    - shadowveryshort
    - near
    - far
    - equal

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.
    spec: Specification name.

    Return:
    --------------------------
    Candle Specification: np.ndarray with preceding np.nan-s.
    """
    realbody = np.abs(close - open_)
    highlow = high - low
    shadows = highlow - realbody
    rgs = {
        "realbody": realbody,
        "highlow": highlow,
        "shadows": shadows,
    }

    settings = {
        "bodylong": ("realbody", 10, 1.0),
        "bodyverylong": ("realbody", 10, 3.0),
        "bodyshort": ("realbody", 10, 1.0),
        "bodydoji": ("highlow", 10, 0.1),
        "shadowlong": ("realbody", 0, 1.0),
        "shadowverylong": ("realbody", 0, 2.0),
        "shadowshort": ("shadows", 10, 1.0),
        "shadowveryshort": ("highlow", 10, 0.1),
        "near": ("highlow", 5, 0.2),
        "far": ("highlow", 5, 0.6),
        "equal": ("highlow", 5, 0.05)
    }

    rg, man, ratio = settings[spec]
    if man == 0:
        ret = ratio * rgs[rg]
    else:
        ret = sma_excur(rgs[rg], man) * ratio

    return ret


# ----------------------------------------------------------------------------
# One candle stick indicator.
# ----------------------------------------------------------------------------
# %%
def belthold(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Candle Stick Belthold.

    1 Candle Pattern:
    --------------------------
    1. Long white or black real body.
    2. No or very short lower(white) or upper(black) shadow.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    BeltHold: np.ndarray filled with 0, 100, -100
    """
    if len(close) < 10:
        logging.warning(f"The number of samples should be larger than {10}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    shadowveryshort = candle_spec(open_, high, low, close, "shadowveryshort")
    bodylong = candle_spec(open_, high, low, close, "bodylong")

    ret = np.zeros_like(close, dtype=np.int_)
    ret[(rb > bodylong)
        & (close > open_)
        & (open_ - low < shadowveryshort)] = 100
    ret[(rb > bodylong)
        & (close < open_)
        & (high - open_ < shadowveryshort)] = -100

    return ret


# %%
def closing_marubozu(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Closing Marubozu.

    1 Candle Pattern:
    --------------------------
    1. Long white or black real body.
    2. No or very short lower(black) or upper(white) shadow.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    Closing Marubozu: np.ndarray filled with 0, 100, -100
    """
    if len(close) < 10:
        logging.warning(f"The number of samples should be larger than {10}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    shadowveryshort = candle_spec(open_, high, low, close, "shadowveryshort")
    bodylong = candle_spec(open_, high, low, close, "bodylong")

    ret = np.zeros_like(close, dtype=np.int_)
    ret[(rb > bodylong)
        & (close > open_)
        & (high - close < shadowveryshort)] = 100
    ret[(rb > bodylong)
        & (close < open_)
        & (close - low < shadowveryshort)] = -100

    return ret


# %%
def doji(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Doji.

    1 Candle Pattern:
    --------------------------
    1. Open quite equal to close.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    Doji: np.ndarray filled with 0, 100
    """
    if len(close) < 10:
        logging.warning(f"The number of samples should be larger than {10}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodydoji = candle_spec(open_, high, low, close, "bodydoji")
    ret = np.zeros_like(close, dtype=np.int_)
    ret[(rb < bodydoji)] = 100

    return ret


# %%
def dragonfly_doji(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Dragonfly Doji.

    1 Candle Pattern:
    --------------------------
    1. Doji body.
    2. No or very short upper shadow.
    3. Not very short lower shadow.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    Dragonfly Doji: np.ndarray filled with 0, 100
    """
    if len(close) < 10:
        logging.warning(f"The number of samples should be larger than {10}, "
                        f"while only {len(close)} samples passed.")
        return None

    shadowveryshort = candle_spec(open_, high, low, close, "shadowveryshort")
    rb = np.abs(close - open_)
    ret = np.zeros_like(close, dtype=np.int_)
    ret[(rb < shadowveryshort)
        & (high - np.maximum(close, open_) < shadowveryshort)
        & (np.minimum(close, open_) - low > shadowveryshort)] = 100

    return ret


# %%
def gravestone_doji(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Gravestone Doji.

    1 Candle Pattern:
    --------------------------
    1. Doji body.
    2. No or very short lower shadow.
    3. Not very short upper shadow.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    Gravestone Doji: np.ndarray filled with 0, 100, -100
    """
    if len(close) < 10:
        logging.warning(f"The number of samples should be larger than {10}, "
                        f"while only {len(close)} samples passed.")
        return None

    shadowveryshort = candle_spec(open_, high, low, close, "shadowveryshort")
    rb = np.abs(close - open_)
    ret = np.zeros_like(close, dtype=np.int_)
    ret[(rb < shadowveryshort)
        & (high - np.maximum(close, open_) > shadowveryshort)
        & (np.minimum(close, open_) - low < shadowveryshort)] = 100

    return ret


# ----------------------------------------------------------------------------
# Two candle stick indicator.
# ----------------------------------------------------------------------------
# %%
def counter_attack(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Counter Attack.

    2 Candle Pattern:
    --------------------------
    1. 1st: Long black or white.
    2. 2nd: Long white(1st black) or black(1st white) with close equal to
      1st's close.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    Counter Attack: np.ndarray filled with 0, 100, -100
    """
    if len(close) < 11:
        logging.warning(f"The number of samples should be larger than {11}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    equal = candle_spec(open_, high, low, close, "equal")
    bodylong = candle_spec(open_, high, low, close, "bodylong")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[1:]
    # Downer bar and then upper bar.
    ret_[(rb[:-1] > bodylong[:-1])
         & (close[:-1] < open_[:-1])
         & (rb[1:] > bodylong[1:])
         & (close[1:] > open_[1:])
         & (np.abs(close[1:] - close[:-1]) < equal[:-1])] = 100
    # Upper bar and then donwer bar.
    ret_[(rb[:-1] > bodylong[:-1])
         & (close[:-1] > open_[:-1])
         & (rb[1:] > bodylong[1:])
         & (close[1:] < open_[1:])
         & (np.abs(close[1:] - close[:-1]) < equal[:-1])] = -100

    return ret


# %%
def darkcloud_cover(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: int = 0.5,
) -> np.ndarray:
    """Counter Attack.

    2 Candle Pattern:
    --------------------------
    1. 1st: Long white.
    2. 2nd: Black opens above the prior day high and closes within prior real
      body.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.
    penetration: The ratio the latter close can't lower than former open.

    Return:
    --------------------------
    DarkCloud Cover: np.ndarray filled with 0, -100
    """
    if len(close) < 11:
        logging.warning(f"The number of samples should be larger than {11}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodylong = candle_spec(open_, high, low, close, "bodylong")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[1:]
    ret_[(rb[:-1] > bodylong[:-1])
         & (open_[:-1] < close[1:])
         & (close[1:] < close[:-1] - rb[:-1] * penetration)
         & (high[:-1] < open_[1:])] = -100

    return ret


# %%
def doji_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Doji Star.

    2 Candle Pattern:
    --------------------------
    1. 1st: Long real body.
    2. 2nd: Star with a doji.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    Doji Star: np.ndarray filled with 0, -100, 100
    """
    if len(close) < 11:
        logging.warning(f"The number of samples should be larger than {11}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodylong = candle_spec(open_, high, low, close, "bodylong")
    bodydoji = candle_spec(open_, high, low, close, "bodydoji")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[1:]
    # Uptrend.
    ret_[(rb[:-1] > bodylong[:-1])
         & (rb[1:] < bodydoji[1:])
         & (open_[:-1] < close[:-1])
         & (close[:-1] < np.minimum(close[1:], open_[1:]))] = -100
    # Downtrend.
    ret_[(rb[:-1] > bodylong[:-1])
         & (rb[1:] < bodydoji[1:])
         & (open_[:-1] > close[:-1])
         & (close[:-1] > np.maximum(close[1:], open_[1:]))] = 100

    return ret


# %%
def engulfing(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Engulf.

    2 Candle Pattern:
    --------------------------
    1. 1st: Black or white.
    2. 2nd: White(1st black) or black(1st white) real body that engulf the
      prior real body.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    Engulfing: np.ndarray filled with 0, -100, 100
    """
    if len(close) < 2:
        logging.warning(f"The number of samples should be larger than {2}, "
                        f"while only {len(close)} samples passed.")
        return None

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[1:]
    ret_[(open_[1:] < close[:-1])
         & (close[:-1] < open_[:-1])
         & (open_[:-1] < close[1:])] = 100
    ret_[(close[1:] < open_[:-1])
         & (open_[:-1] < close[:-1])
         & (close[:-1] < open_[1:])] = -100
    # The second element can't be set, but why?
    ret[:2] = 0

    return ret


# %%
def hammer(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Hammer.

    2 Candle Pattern:
    --------------------------
    1. 2nd: Small real body.
    2. 2nd: Long lower shadow.
    3. 2nd: No or very short upper shadow.
    4. 2nd: Body below or near the lows of the prior.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    Hammer: np.ndarray filled with 0, 100
    """
    if len(close) < 12:
        logging.warning(f"The number of samples should be larger than {12}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodyshort = candle_spec(open_, high, low, close, "bodyshort")
    shadowlong = candle_spec(open_, high, low, close, "shadowlong")
    shadowveryshort = candle_spec(open_, high, low, close, "shadowveryshort")
    near = candle_spec(open_, high, low, close, "near")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[1:]
    ret_[(rb[1:] < bodyshort[1:])
         & (np.minimum(close, open_)[1:] - low[1:] > shadowlong[1:])
         & (high[1:] - np.maximum(close, open_)[1:] < shadowveryshort[1:])
         & (np.minimum(close, open_)[1:] <= low[:-1] + near[:-1])] = 100
    # The 11th element can't be set, but why?
    ret[:11] = 0

    return ret


# %%
def hangingman(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Hangingman.

    2 Candle Pattern:
    --------------------------
    1. 2nd: Small real body.
    2. 2nd: Long lower shadow.
    3. 2nd: No or very short upper shadow.
    4. 2nd: Body above or near the highs of the prior.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    Hangingman: np.ndarray filled with 0, -100
    """
    if len(close) < 12:
        logging.warning(f"The number of samples should be larger than {12}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodyshort = candle_spec(open_, high, low, close, "bodyshort")
    shadowlong = candle_spec(open_, high, low, close, "shadowlong")
    shadowveryshort = candle_spec(open_, high, low, close, "shadowveryshort")
    near = candle_spec(open_, high, low, close, "near")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[1:]
    ret_[(rb[1:] < bodyshort[1:])
         & (np.minimum(close, open_)[1:] - low[1:] > shadowlong[1:])
         & (high[1:] - np.maximum(close, open_)[1:] < shadowveryshort[1:])
         & (np.minimum(close, open_)[1:] >= high[:-1] - near[:-1])] = -100
    # The 11th element can't be set, but why?
    ret[:11] = 0

    return ret


# %%
def crows2(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """2 Crows.

    3 Candle Pattern:
    --------------------------
    1. 1st: Long white body.
    2. 2nd: Black real body.
    3. 1st 2nd: Gap between 1st and 2nd real bodies.
    4. 3rd: Black open within 2nd real body and close within the 1st real body.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    2Crows: np.ndarray filled with 0, -100
    """
    if len(close) < 12:
        logging.warning(f"The number of samples should be larger than {12}, "
                        f"while only {len(close)} samples passed.")
        return None

    bodylong = candle_spec(open_, high, low, close, "bodylong")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[2:]
    # ret_[(open_[:-2] + bodylong[:-2] < close[:-2])
    #      & (close[:-2] < close[1:-1])
    #      & (close[1:-1] < open_[2:])
    #      & (open_[2:] < open_[1:-1])
    #      & (close[2:] < close[:-2]) & (close[2:] > open_[:-2])] = -100
    ret_[(open_[:-2] + bodylong[:-2] < close[:-2])
         & (open_[1:-1] > close[1:-1])
         & (close[:-2] < close[1:-1])
         # & (open_[2:] > close[2:])
         & (open_[2:] < open_[1:-1]) & (open_[2:] > close[1:-1])
         & (close[2:] < close[:-2]) & (close[2:] > open_[:-2])] = -100

    return ret


# %%
def black_crows3(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """3 Black Crows.

    4 Candle Pattern:
    --------------------------
    1. 1st: White and 3 consecutive and declining black candle.
    2. 2nd, 3rd, 4th: No or very short lower shadow.
    3. 3rd, 4th: Open within the prior real body.
    4. 2nd: Close should be under the prior high.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    3 Black Crows: np.ndarray filled with 0, -100
    """
    if len(close) < 13:
        logging.warning(f"The number of samples should be larger than {13}, "
                        f"while only {len(close)} samples passed.")
        return None

    shadowveryshort = candle_spec(open_, high, low, close, "shadowveryshort")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[3:]
    # 1. First upper candle and 3 consecutive and declining downer candle.
    ret_[(close[:-3] > open_[:-3])
         & (close[1:-2] < open_[1:-2])
         & (close[2:-1] < open_[2:-1])
         & (close[3:] < open_[3:])
         & (close[1:-2] > close[2:-1])
         & (close[2:-1] > close[3:])
         # 2. 2nd, 3rd, 4th must have no or very short lower shadow.
         & (close[1:-2] - low[1:-2] < shadowveryshort[1:-2])
         & (close[2:-1] - low[2:-1] < shadowveryshort[2:-1])
         & (close[3:] - low[3:] < shadowveryshort[3:])
         # 3. 3rd, 4th must open within the prior real body.
         & (open_[2:-1] < open_[1:-2]) & (open_[2:-1] > close[1:-2])
         & (open_[3:] < open_[2:-1]) & (open_[3:] > close[2:-1])
         # 4. 2nd's close should be under the 1nd's high.
         & (close[1:-2] < high[:-3])] = -100
    # ????
    ret[:13] = 0

    return ret


# %%
def inside3(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """3 Inside.

    3 Candle Pattern:
    --------------------------
    1. 1st: Long white(black) body.
    2. 2nd: Short real body engulfed by the prior body.
    3. 3rd: Black(white) closes lower(higher) than 1st's open.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    3 Inside: np.ndarray filled with 0, -100
    """
    if len(close) < 13:
        logging.warning(f"The number of samples should be larger than {13}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodylong = candle_spec(open_, high, low, close, "bodylong")
    bodyshort = candle_spec(open_, high, low, close, "bodyshort")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[2:]
    ret_[(open_[:-2] + bodylong[:-2] < close[:-2])
         & (rb[1:-1] < bodyshort[1:-1])
         & (np.maximum(open_[1:-1], close[1:-1]) < close[:-2])
         & (np.minimum(open_[1:-1], close[1:-1]) > open_[:-2])
         & (open_[2:] > close[2:])
         & (close[2:] < open_[:-2])] = -100
    ret_[(close[:-2] + bodylong[:-2] < open_[:-2])
         & (rb[1:-1] < bodyshort[1:-1])
         & (np.maximum(open_[1:-1], close[1:-1]) < open_[:-2])
         & (np.minimum(open_[1:-1], close[1:-1]) > close[:-2])
         & (open_[2:] < close[2:])
         & (close[2:] > open_[:-2])] = 100

    return ret


# %%
def outside3(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """3 Outside.

    3 Candle Pattern:
    --------------------------
    1. 1st: Black(white) body.
    2. 2nd: White(black) body engulfs the prior.
    3. 3rd: Closes higher(lower) than 2nd's close.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    3 Outside: np.ndarray filled with 0, -100
    """
    if len(close) < 3:
        logging.warning(f"The number of samples should be larger than {3}, "
                        f"while only {len(close)} samples passed.")
        return None

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[2:]
    ret_[(open_[:-2] > close[:-2])
         & (open_[1:-1] < close[:-2])
         & (close[1:-1] > open_[:-2])
         & (close[2:] > close[1:-1])] = 100
    ret_[(open_[:-2] < close[:-2])
         & (open_[1:-1] > close[:-2])
         & (close[1:-1] < open_[:-2])
         & (close[2:] < close[1:-1])] = -100
    # ????
    ret[:3] = 0

    return ret


# %%
def stars_insouth3(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """3 Stars in South.

    3 Candle Pattern:
    --------------------------
    1. 1st: Long black with long lower shadow.
    2. 2nd: Smaller(than prior) black opens higher than prior close but
      within prior range(not body).
    3. 2nd: Trades lower than prior close but not lower than prior low.
    4. 2nd: Closes off its low(has lower shadow).
    5. 3rd: Small black marubozu engulfed by prior range.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    3 Stars In South: np.ndarray filled with 0, -100
    """
    if len(close) < 3:
        logging.warning(f"The number of samples should be larger than {3}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodylong = candle_spec(open_, high, low, close, "bodylong")
    bodyshort = candle_spec(open_, high, low, close, "bodyshort")
    shadowlong = candle_spec(open_, high, low, close, "shadowlong")
    shadowveryshort = candle_spec(open_, high, low, close, "shadowveryshort")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[2:]
    # 1. 1st: Long black with long lower shadow.
    ret_[(close[:-2] + bodylong[:-2] < open_[:-2])
         & (close[:-2] - low[:-2] > shadowlong[:-2])
         # 2. 2nd: Smaller black opens higher than prior close but within
         # prior range(not body).
         & (rb[1:-1] < rb[:-2])
         & (open_[1:-1] > close[1:-1])
         & (open_[1:-1] > close[:-2])
         & (open_[1:-1] < high[:-2])
         # 3. 2nd: Trades lower than prior close but not lower than prior low.
         & (close[1:-1] < close[:-2])
         & (close[1:-1] > low[:-2])
         # 4. 2nd: Closes off its low(has lower shadow).
         & (close[1:-1] - low[1:-1] > shadowveryshort[1:-1])
         # 5. 3rd: Small black marubozu engulfed by prior range.
         & (rb[2:] < bodyshort[2:])
         & (close[2:] < open_[2:])
         & (high[2:] - open_[2:] < shadowveryshort[2:])
         & (close[2:] - low[2:] < shadowveryshort[2:])
         & (low[2:] > low[1:-1])
         & (high[2:] < high[1:-1])] = 100

    return ret


# %%
def white_soldiers3(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """3 White Soldiers.

    3 Candle Pattern:
    --------------------------
    1. 1st, 2nd, 3rd: White with consecutively higher closes.
    2. 1st, 2nd, 3rd: Not short.
    3. 1st, 2nd, 3rd: Opens within or near prior real body.
    4. 1st, 2nd, 3rd: No or very short upper shadow.
    5. 1st, 2nd, 3rd: Not far shorter than prior.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    3 White Soldiers: np.ndarray filled with 0, -100
    """
    if len(close) < 3:
        logging.warning(f"The number of samples should be larger than {3}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodyshort = candle_spec(open_, high, low, close, "bodyshort")
    near = candle_spec(open_, high, low, close, "near")
    far = candle_spec(open_, high, low, close, "far")
    shadowveryshort = candle_spec(open_, high, low, close, "shadowveryshort")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[2:]
    # 1. 1st, 2nd, 3rd: White with consecutively higher closes.
    ret_[(open_[:-2] < close[:-2])
         & (open_[1:-1] < close[1:-1])
         & (open_[2:] < close[2:])
         & (close[:-2] < close[1:-1])
         & (close[1:-1] < close[2:])
         # 2. 1st, 2nd, 3rd: Not short.
         & (rb[:-2] > bodyshort[:-2])
         & (rb[1:-1] > bodyshort[1:-1])
         & (rb[2:] > bodyshort[2:])
         # 3. 1st, 2nd, 3rd: Opens within or near prior real body.
         & (open_[1:-1] > open_[:-2])
         & (open_[1:-1] < close[:-2] + near[:-2])
         & (open_[2:] > open_[1:-1])
         & (open_[2:] < close[1:-1] + near[1:-1])
         # 4. 1st, 2nd, 3rd: No or very short upper shadow.
         & (high[:-2] - close[:-2] < shadowveryshort[:-2])
         & (high[1:-1] - close[1:-1] < shadowveryshort[1:-1])
         & (high[2:] - close[2:] < shadowveryshort[2:])
         # 5. 1st, 2nd, 3rd: Not far shorter than prior.
         & (rb[1:-1] > rb[:-2] - far[:-2])
         & (rb[2:] > rb[1:-1] - far[1:-1])] = 100

    return ret


# %%
def line_strike3(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """3 Line Strike.

    4 Candle Pattern:
    --------------------------
    1. 1st, 2nd, 3rd: Three white soldiers or three black crows.
      Three whites(blacks) with consecutively higher(lower) closes opens
      within or near the prior real body.
    2. 4th: Black(white) opens above(below) prior close and closes below(above)
      1st open.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.
    penetration: The ratio for overlapping of candles.

    Return:
    --------------------------
    2 Line Strike: np.ndarray filled with 0, -100
    """
    if len(close) < 14:
        logging.warning(f"The number of samples should be larger than {14}, "
                        f"while only {len(close)} samples passed.")
        return None

    near = candle_spec(open_, high, low, close, "near")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[3:]
    # 1. 1st, 2nd, 3rd: Three white soldiers or three black crows.
    ret_[(close[:-3] > open_[:-3])
         & (close[1:-2] > open_[1:-2])
         & (close[2:-1] > open_[2:-1])
         & (close[:-3] < close[1:-2])
         & (close[1:-2] < close[2:-1])
         & (open_[1:-2] < np.maximum(close[:-3], open_[:-3]) + near[:-3])
         & (open_[1:-2] > np.minimum(close[:-3], open_[:-3]) - near[:-3])
         & (open_[2:-1] < np.maximum(close[1:-2], open_[1:-2]) + near[1:-2])
         & (open_[2:-1] > np.minimum(close[1:-2], open_[1:-2]) - near[1:-2])
         # 2. 4th: Black(white) opens above(below) prior close and closes
         # below(above) 1st open.
         & (open_[3:] > close[2:-1])
         & (close[3:] < open_[:-3])] = 100
    # 1. 1st, 2nd, 3rd: Three white soldiers or three black crows.
    ret_[(close[:-3] < open_[:-3])
         & (close[1:-2] < open_[1:-2])
         & (close[2:-1] < open_[2:-1])
         & (close[:-3] > close[1:-2])
         & (close[1:-2] > close[2:-1])
         & (open_[1:-2] < np.maximum(close[:-3], open_[:-3]) + near[:-3])
         & (open_[1:-2] > np.minimum(close[:-3], open_[:-3]) - near[:-3])
         & (open_[2:-1] < np.maximum(close[1:-2], open_[1:-2]) + near[1:-2])
         & (open_[2:-1] > np.minimum(close[1:-2], open_[1:-2]) - near[1:-2])
         # 2. 4th: Black(white) opens above(below) prior close and closes
         # below(above) 1st open.
         & (open_[3:] < close[2:-1])
         & (close[3:] > open_[:-3])] = -100

    return ret


# %%
def conceal_baby_swall(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """3 Line Strike.

    4 Candle Pattern:
    --------------------------
    1. 1st, 2nd: Black marubozu.
    2. 3rd: Black opens gapping down with upper shadow extends to
      prior real body.
    1. 4st: Black engulf the range of 3rd.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.
    penetration: The ratio for overlapping of candles.

    Return:
    --------------------------
    Conceal Baby swall: np.ndarray filled with 0, -100
    """
    if len(close) < 14:
        logging.warning(f"The number of samples should be larger than {14}, "
                        f"while only {len(close)} samples passed.")
        return None

    shadowveryshort = candle_spec(open_, high, low, close, "shadowveryshort")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[3:]
    # 1. 1st, 2nd: Black marubozu.
    ret_[(close[:-3] < open_[:-3])
         & (close[:-3] - low[:-3] < shadowveryshort[:-3])
         & (high[:-3] - open_[:-3] < shadowveryshort[:-3])
         & (close[1:-2] < open_[1:-2])
         & (close[1:-2] - low[1:-2] < shadowveryshort[1:-2])
         & (high[1:-2] - open_[1:-2] < shadowveryshort[1:-2])
         # 2. 3rd: Black opens gapping down with upper shadow extends to
         # prior real body.
         & (close[2:-1] < open_[2:-1])
         & (open_[2:-1] < close[1:-2])
         & (high[2:-1] > close[1:-2])
         # 3. 4st: Black engulf the range of 3rd.
         & (open_[3:] > high[2:-1])
         & (close[3:] < low[2:-1])] = 100

    return ret


# %%
def abandoned_baby(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float = 0.3,
) -> np.ndarray:
    """Abandoned Baby.

    3 Candle Pattern:
    --------------------------
    1. 1st: Long white(black) real body.
    2. 2nd: Doji.
    3. 3rd: Not short black(white) real body moves well in 1st's real body.
    4. 1st, 2nd: 2nd gap up(down) 1st with shadows not overlaping.
    5. 2rd, 3nd: 3nd gap down(up) 2rd with shadows not overlaping.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.
    penetration: The ratio for overlapping of candles.

    Return:
    --------------------------
    Abandoned Baby: np.ndarray filled with 0, -100
    """
    if len(close) < 3:
        logging.warning(f"The number of samples should be larger than {3}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodylong = candle_spec(open_, high, low, close, "bodylong")
    bodyshort = candle_spec(open_, high, low, close, "bodyshort")
    bodydoji = candle_spec(open_, high, low, close, "bodydoji")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[2:]
    ret_[(open_[:-2] + bodylong[:-2] < close[:-2])
         & (rb[1:-1] < bodydoji[1:-1])
         & (open_[2:] > close[2:] + bodyshort[2:])
         & (close[2:] > close[:-2] - rb[:-2] * penetration)
         & (low[1:-1] > high[:-2])
         & (low[1:-1] > high[2:])] = -100
    ret_[(close[:-2] + bodylong[:-2] < open_[:-2])
         & (rb[1:-1] < bodydoji[1:-1])
         & (close[2:] > open_[2:] + bodyshort[2:])
         & (close[2:] > close[:-2] - rb[:-2] * penetration)
         & (high[1:-1] < low[:-2])
         & (high[1:-1] < low[2:])] = 100

    return ret


# %%
def evening_doji_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float = 0.3,
) -> np.ndarray:
    """Evening Doji Star.

    3 Candle Pattern:
    --------------------------
    1. 1st: Long white real body.
    2. 2nd: Doji real body gapping above prior real body.
    3. 3rd: Black not short real body moves well within the 1st's real body.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.
    penetration: The ratio for overlapping of candles.

    Return:
    --------------------------
    Evening Doji Star: np.ndarray filled with 0, -100
    """
    if len(close) < 3:
        logging.warning(f"The number of samples should be larger than {3}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodylong = candle_spec(open_, high, low, close, "bodylong")
    bodyshort = candle_spec(open_, high, low, close, "bodyshort")
    bodydoji = candle_spec(open_, high, low, close, "bodydoji")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[2:]
    # 1. 1st: Long white real body.
    ret_[(open_[:-2] + bodylong[:-2] < close[:-2])
         # 2. 2nd: Doji real body gapping above prior real body.
         & (rb[1:-1] < bodydoji[1:-1])
         & (np.minimum(open_[1:-1], close[1:-1]) > np.maximum(open_[:-2], close[:-2]))
         # 3. 3rd: Black not short real body moves well within the 1st's real body.
         & (rb[2:] > bodyshort[2:])
         & (open_[2:] > close[2:])
         & (close[2:] < close[:-2] - rb[:-2] * penetration)] = -100

    return ret


# %%
def evening_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float = 0.3,
) -> np.ndarray:
    """Evening Star.

    3 Candle Pattern:
    --------------------------
    1. 1st: Long white real body.
    2. 2nd: Star(short real body gapping above prior real body).
    3. 3rd: Black not short real body moves well within the 1st's real body.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.
    penetration: The ratio for overlapping of candles.

    Return:
    --------------------------
    Evening Star: np.ndarray filled with 0, -100
    """
    if len(close) < 3:
        logging.warning(f"The number of samples should be larger than {3}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodylong = candle_spec(open_, high, low, close, "bodylong")
    bodyshort = candle_spec(open_, high, low, close, "bodyshort")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[2:]
    # 1. 1st: Long white real body.
    ret_[(open_[:-2] + bodylong[:-2] < close[:-2])
         # 2. 2nd: Doji real body gapping above prior real body.
         & (rb[1:-1] < bodyshort[1:-1])
         & (np.minimum(open_[1:-1], close[1:-1]) > np.maximum(open_[:-2], close[:-2]))
         # 3. 3rd: Black not short real body moves well within the 1st's real body.
         & (rb[2:] > bodyshort[2:])
         & (open_[2:] > close[2:])
         & (close[2:] < close[:-2] - rb[:-2] * penetration)] = -100

    return ret


# %%
def gap_side_side_white(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    penetration: float = 0.3,
) -> np.ndarray:
    """Gap Side Side White.

    3 Candle Pattern:
    --------------------------
    1. 2nd, 3rd: Two whites of the near size and about the equal open.
    2. 2nd, 3rd: Two real bodies gap above or below the 1st real body.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.
    penetration: The ratio for overlapping of candles.

    Return:
    --------------------------
    Go Side Side White: np.ndarray filled with 0, -100
    """
    if len(close) < 7:
        logging.warning(f"The number of samples should be larger than {7}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    near = candle_spec(open_, high, low, close, "near")
    equal = candle_spec(open_, high, low, close, "equal")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[2:]
    # 1. 2nd, 3rd: White of the near size and about the equal open.
    ret_[(close[1:-1] > open_[1:-1])
         & (close[2:] > open_[2:])
         & (np.abs(rb[1:-1] - rb[2:]) < near[1:-1])
         & (np.abs(open_[1:-1] - open_[2:]) < equal[1:-1])
         # 2. 2nd, 3rd: Real body gap above or below the 1st real body.
         & (open_[1:-1] > np.maximum(open_[:-2], close[:-2]))
         & (open_[2:] > np.maximum(open_[:-2], close[:-2]))] = 100
    ret_[(close[1:-1] > open_[1:-1])
         & (close[2:] > open_[2:])
         & (np.abs(rb[1:-1] - rb[2:]) < near[1:-1])
         & (np.abs(open_[1:-1] - open_[2:]) < equal[1:-1])
         & (close[1:-1] < np.minimum(open_[:-2], close[:-2]))
         & (close[2:] < np.minimum(open_[:-2], close[:-2]))] = -100
    # ??????
    ret[:7] = 0

    return ret


# %%
def break_away(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Break Away.

    5 Candle Pattern:
    --------------------------
    1. 1st: Long black.
    2. 2nd: Black(white) body gaps down(up).
    3. 3rd, 4th: Black(white) with lower(higher) high and lower(higher) low than prior.
    4. 5th: White(black) closes inside the gap of 1st and 2nd.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    Break Away: np.ndarray filled with 0, -100
    """
    if len(close) < 3:
        logging.warning(f"The number of samples should be larger than {3}, "
                        f"while only {len(close)} samples passed.")
        return None

    bodylong = candle_spec(open_, high, low, close, "bodylong")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[4:]
    # 1. 1st: Long black.
    ret_[(close[:-4] + bodylong[:-4] < open_[:-4])
         # 2. 2nd: Black(white) body gaps down(up).
         & (close[1:-3] < open_[1:-3])
         & (open_[1:-3] < close[:-4])
         # 3. 3rd, 4th: Black(white) with lower(higher) high and lower(higher)
         # low than prior.
         & (close[2:-2] < open_[2:-2])
         & (low[2:-2] < low[1:-3])
         & (high[2:-2] < high[1:-3])
         & (close[3:-1] < open_[3:-1])
         & (low[3:-1] < low[2:-2])
         & (high[3:-1] < high[2:-2])
         # 4. 5th: White(black) closes inside the gap of 1st and 2nd.
         & (close[4:] > open_[4:])
         & (close[4:] > open_[1:-3])
         & (close[4:] < close[:-4])] = 100
    # 1. 1st: Long black.
    ret_[(close[:-4] + bodylong[:-4] < open_[:-4])
         # 2. 2nd: Black(white) body gaps down(up).
         & (close[1:-3] > open_[1:-3])
         & (open_[1:-3] > open_[:-4])
         # 3. 3rd, 4th: Black(white) with lower(higher) high and lower(higher)
         # low than prior.
         & (close[2:-2] > open_[2:-2])
         & (low[2:-2] > low[1:-3])
         & (high[2:-2] > high[1:-3])
         & (close[3:-1] > open_[3:-1])
         & (low[3:-1] > low[2:-2])
         & (high[3:-1] > high[2:-2])
         # 4. 5th: White(black) closes inside the gap of 1st and 2nd.
         & (close[4:] < open_[4:])
         & (close[4:] < open_[1:-3])
         & (close[4:] > open_[:-4])] = 100

    return ret

# %%
# TODO: Can't pass tests.
def advance_block(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Advance Block.

    3 Candle Pattern:
    --------------------------
    1. 1st, 2nd, 3rd: White with consecutively higher closes.
    2. 2nd, 3rd: Opens within or near(above) the prior white real body.
    3. 1st: Long white with no or short upper shadow.
    4. 2nd, 3rd: Sign of weakening
      - Progressively smaller white real body.
      - Relatively long upper shadows.

    Params:
    --------------------------
    open_: Open price.
    high: High price.
    low: Low price.
    close: Close price.

    Return:
    --------------------------
    Advance Block: np.ndarray filled with 0, -100
    """
    if len(close) < 3:
        logging.warning(f"The number of samples should be larger than {3}, "
                        f"while only {len(close)} samples passed.")
        return None

    rb = np.abs(close - open_)
    bodylong = candle_spec(open_, high, low, close, "bodylong")
    near = candle_spec(open_, high, low, close, "near")
    far = candle_spec(open_, high, low, close, "far")
    shadowshort = candle_spec(open_, high, low, close, "shadowshort")
    shadowlong = candle_spec(open_, high, low, close, "shadowlong")

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[2:]
    # 1. 1st, 2nd, 3rd: White with consecutively higher closes.
    ret_[(close[:-2] > open_[:-2])
         & (close[1:-1] > open_[1:-1])
         & (close[2:] > open_[2:])
         & (close[2:] > close[1:-1])
         & (close[1:-1] > close[:-2])
         # 2. 2nd, 3rd: Opens within or near(above) the prior white real body.
         & (open_[1:-1] > open_[:-2])
         & (open_[1:-1] <= close[:-2] + near[:-2])
         & (open_[2:] > open_[1:-1])
         & (open_[2:] <= close[1:-1] + near[1:-1])
         # 3. 1st: Long white with no or short upper shadow.
         & (rb[:-2] > bodylong[:-2])
         & (high[:-2] - close[:-2] < shadowshort[:-2])
         # 4. 2nd, 3rd: Sign of weakening.
         # 4.1 2nd far smaller than 1st and 3rd not near longer than 2nd.
         & (((rb[1:-1] < rb[:-2] - far[:-2])
             & (rb[2:] < rb[1:-1] + near[1:-1]))
            # 4.2 3rd far smaller than 2nd.
            | (rb[2:] < rb[1:-1] - far[1:-1])
            # 4.3 3rd smaller than 2nd, 2nd smaller than 1st and 3rd or 2nd
            #   with not short upper shadow.
            | ((rb[2:] < rb[1:-1])
               & (rb[1:-1] < rb[:-2])
               & (((high[2:] - close[2:] > shadowshort[2:]))
                  | (high[1:-1] - close[1:-1] > shadowshort[1:-1])))
            # 4.4 3rd smaller than 2nd and 3rd with long upper shadow.
            | ((rb[2:] < rb[1:-1])
               & (high[2:] - close[2:] > shadowlong[2:])))] = -100

    return ret
