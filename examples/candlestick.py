#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: kline.py
#   Author: xyy15926
#   Created: 2025-01-20 18:02:59
#   Updated: 2025-01-20 18:03:51
#   Description:
# ---------------------------------------------------------

# %%
import json
import talib as ta
import pandas as pd

from ringbear.talib.overlap import ma
from ringbear.talib.momentum import aroon
from ringbear.talib.candlestick import advance_block

from ubears.flagbear.slp.finer import tmp_file, get_assets_path
from ubears.suitbear.candlestick import draw_kline

# %%
if __name__ == "__main__":
    fnamep = "stock/kline.html"
    fname = tmp_file(fnamep, incr=0).with_suffix(".html")
    dfile = get_assets_path() / "stock/stock_jdi.json"
    data = json.load(open(dfile))
    data = pd.DataFrame.from_records(
        data,
        columns=["date", "open_", "close", "low", "high", "volume"])
    mas = pd.DataFrame({
        "MA30": ma(data["close"].values),
    })
    ptns = pd.DataFrame({
        "AROON14": aroon(data["high"].values, data["low"].values)[0],
        "ADB": advance_block(data["open_"].values, data["high"].values,
                             data["low"].values, data["close"].values),
        "TAADB": ta.CDLADVANCEBLOCK(data["open_"], data["high"],
                                    data["low"], data["close"]),
    })
    ka = draw_kline(data, mas, ptns, fname)
