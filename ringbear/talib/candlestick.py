#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: candlestick.py
#   Author: xyy15926
#   Created: 2024-11-25 13:44:52
#   Updated: 2024-11-26 15:22:50
#   Description:
#   Ref: https://github.com/frgomes/ta-lib_code/blob/master/ta-lib/c/src/ta_func/
#   Ref: https://www.fmlabs.com/reference/default.htm
#   Ref: https:https://blog.csdn.net/weixin_43420026/article/details/126743440
#   Ref: https://blog.csdn.net/suiyingy/article/details/118661718
#   Global Setting: https://github.com/frgomes/ta-lib_code/blob/master/ta-lib/c/src/ta_common/ta_global.c#L128
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

    Just like shift the original SMA 1 step.

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
def candle_how(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    how: str,
) -> np.ndarray:
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

    rg, man, ratio = settings[how]
    if man == 0:
        ret = ratio * rgs[rg]
    else:
        ret = sma_excur(rgs[rg], man) * ratio

    return ret


# %%
def belthold(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Candle Stick Belthold.

    1. Long body.
    2. Very short lower shadow.

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
    MAN = 10
    if len(close) < MAN:
        logging.warning(f"The number of samples should be larger than {MAN}, "
                        f"while only {len(close)} samples passed.")
        return None

    # Use MA without current bar here.
    shadow_veryshort = sma_excur(high - low, MAN) * 0.1
    body_real = np.abs(close - open_)
    body_long = sma_excur(body_real, MAN)
    ret = np.zeros_like(close, dtype=np.int_)
    ret[(body_real > body_long)
        & (close > open_)
        & (shadow_veryshort > (open_ - low))] = 100
    ret[(body_real > body_long)
        & (close < open_)
        & (shadow_veryshort > (high - open_))] = -100

    return ret


# %%
def closing_marubozu(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Closing Marubozu.

    1. Long body.
    2. Very short upper shadow.

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
    MAN = 10
    if len(close) < MAN:
        logging.warning(f"The number of samples should be larger than {MAN}, "
                        f"while only {len(close)} samples passed.")
        return None

    # Use MA without current bar here.
    shadow_veryshort = sma_excur(high - low, MAN) * 0.1
    body_real = np.abs(close - open_)
    body_long = sma_excur(body_real, MAN)

    ret = np.zeros_like(close, dtype=np.int_)
    ret[(body_real > body_long)
        & (close > open_)
        & (shadow_veryshort > (high - close))] = 100
    ret[(body_real > body_long)
        & (close < open_)
        & (shadow_veryshort > (close - low))] = -100

    return ret


# %%
def doji(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Doji.

    1. Very short body.

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
    MAN = 10
    if len(close) < MAN:
        logging.warning(f"The number of samples should be larger than {MAN}, "
                        f"while only {len(close)} samples passed.")
        return None

    # Here, MA excludes current bar.
    shadow_veryshort = sma_excur(high - low, MAN) * 0.1
    body_real = np.abs(close - open_)
    ret = np.zeros_like(close, dtype=np.int_)
    ret[(body_real < shadow_veryshort)] = 100

    return ret


# %%
def dragonfly_doji(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Dragonfly Doji.

    1. Very short body.
    2. Very short upper shadow.
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
    MAN = 10
    if len(close) < MAN:
        logging.warning(f"The number of samples should be larger than {MAN}, "
                        f"while only {len(close)} samples passed.")
        return None

    # Here, MA excludes current bar.
    shadow_veryshort = sma_excur(high - low, MAN) * 0.1
    body_real = np.abs(close - open_)
    ret = np.zeros_like(close, dtype=np.int_)
    ret[(body_real < shadow_veryshort)
        & (high - np.maximum(close, open_) < shadow_veryshort)
        & (np.minimum(close, open_) - low > shadow_veryshort)] = 100

    return ret


# %%
def gravestone_doji(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Gravestone Doji.

    1. Very short body.
    2. Very short upper shadow.
    3. Not very short lower shadow.

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
    MAN = 10
    if len(close) < MAN:
        logging.warning(f"The number of samples should be larger than {MAN}, "
                        f"while only {len(close)} samples passed.")
        return None

    shadow_veryshort = sma_excur(high - low, MAN) * 0.1
    body_real = np.abs(close - open_)
    ret = np.zeros_like(close, dtype=np.int_)
    ret[(body_real < shadow_veryshort)
        & (high - np.maximum(close, open_) > shadow_veryshort)
        & (np.minimum(close, open_) - low < shadow_veryshort)] = 100

    return ret


# %%
def counter_attack(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """Counter Attack.

    In adjacent two days,
    1. Upper bar or downer bar following the other.
    2. Closes are nearly equal.
    3. Both body are long.

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

    shadow_equal = sma_excur(high - low, 5) * 0.05
    body_real = np.abs(close - open_)
    body_long = sma_excur(body_real, 10)

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[1:]
    # Downer bar and then upper bar.
    ret_[(body_real[:-1] > body_long[:-1])
         & (body_real[1:] > body_long[1:])
         & (close[:-1] < open_[:-1])
         & (close[1:] > open_[1:])
         & (np.abs(close[1:] - close[:-1]) < shadow_equal[:-1])] = 100
    # Upper bar and then donwer bar.
    ret_[(body_real[:-1] > body_long[:-1])
         & (body_real[1:] > body_long[1:])
         & (close[:-1] > open_[:-1])
         & (close[1:] < open_[1:])
         & (np.abs(close[1:] - close[:-1]) < shadow_equal[:-1])] = -100

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

    In adjacent two days,
    1. Former bar body are long.
    2. Downer bar follows the upper bar.
    3. Latter open is higher than former high.
    4. Latter close is lower than former close but higher than former open.

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

    body_real = np.abs(close - open_)
    body_long = sma_excur(body_real, 10)

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[1:]
    ret_[(body_real[:-1] > body_long[:-1])
         & (open_[:-1] < close[1:])
         & (close[1:] < close[:-1] - body_real[:-1] * penetration)
         & (high[:-1] < open_[1:])] = -100

    return ret


# %%
def doji_star(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Counter Attack.

    In adjacent two days,
    1. Former bar body is long and latter bar body is very short.
    2. The latter bar keep the trend with the former bar almostly.
      2.1 Upper former bar -> min(close, open) > close[-1]
      2.2 Donwer former bar -> max(close, open) < close[-1]

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

    body_real = np.abs(close - open_)
    body_long = sma_excur(body_real, 10)
    shadow_veryshort = sma_excur(high - low, 10) * 0.1

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[1:]
    ret_[(body_real[:-1] > body_long[:-1])
         & (body_real[1:] < shadow_veryshort[1:])
         & (open_[:-1] < close[:-1])
         & (close[:-1] < np.minimum(close[1:], open_[1:]))] = -100
    ret_[(body_real[:-1] > body_long[:-1])
         & (body_real[1:] < shadow_veryshort[1:])
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

    In adjacent two days,
    1. The latter bar engulfs the the former bar strictly.

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
        logging.warning(f"The number of samples should be larger than {11}, "
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

    For the latter candle,
    1. Small real body.
    2. Long lower shadow.
    3. No or very short upper shadow.
    4. Body below or near the lows of the previous candle.

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
    if len(close) < 10:
        logging.warning(f"The number of samples should be larger than {10}, "
                        f"while only {len(close)} samples passed.")
        return None

    shadow_veryshort = candle_how(open_, high, low, close, "shadowveryshort")
    shadow_near = candle_how(open_, high, low, close, "near")
    body_real = np.abs(close - open_)
    body_long = sma_excur(body_real, 10)

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[1:]
    ret_[(body_real[1:] < body_long[1:])
         & (np.minimum(close, open_)[1:] - low[1:] > body_real[1:])
         & (high[1:] - np.maximum(close, open_)[1:] < shadow_veryshort[1:])
         & (np.minimum(close, open_)[1:] < low[:-1] + shadow_near[:-1])] = 100
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

    For the latter candle,
    1. Small real body.
    2. Long lower shadow.
    3. No or very short upper shadow.
    4. Body above or near the highs of the previous candle.

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
    if len(close) < 10:
        logging.warning(f"The number of samples should be larger than {10}, "
                        f"while only {len(close)} samples passed.")
        return None

    shadow_veryshort = candle_how(open_, high, low, close, "shadowveryshort")
    shadow_near = candle_how(open_, high, low, close, "near")
    body_real = np.abs(close - open_)
    body_long = sma_excur(body_real, 10)

    ret = np.zeros_like(close, dtype=np.int_)
    ret_ = ret[1:]
    ret_[(body_real[1:] < body_long[1:])
         & (np.minimum(close, open_)[1:] - low[1:] > body_real[1:])
         & (high[1:] - np.maximum(close, open_)[1:] < shadow_veryshort[1:])
         & (np.minimum(close, open_)[1:] > high[:-1] - shadow_near[:-1])] = -100
    # The 11th element can't be set, but why?
    ret[:11] = 0

    return ret
