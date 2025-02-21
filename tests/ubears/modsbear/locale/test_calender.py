#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_calender.py
#   Author: xyy15926
#   Created: 2025-02-20 15:33:21
#   Updated: 2025-02-21 11:13:55
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ubears.modsbear.locale import calender
    reload(calender)

import numpy as np
import pandas as pd
from chinese_calendar import holidays
from ubears.modsbear.locale.calender import (
    ChnBusdayCalendar,
    is_chn_busday,
    not_chn_busday,
    ChineseHolidaysCalendar
)


# %%
def test_is_chn_busday():
    holis = np.asarray(list(holidays.keys()), dtype="M8[D]")
    assert np.all(ChnBusdayCalendar.weekmask == [1, 1, 1, 1, 1, 0, 0])
    # `np.busdaycalender` will drop weekends in passed-in holidays
    # automatically.
    assert len(holis) > len(ChnBusdayCalendar.holidays)
    assert np.all(~is_chn_busday(holis))

    x = ["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"]
    ret = is_chn_busday(x)
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = not_chn_busday(x)
    assert np.all(ret == [0, 1, 1, 1, 0])

    x = np.array(["2021-11-11", "2025-01-01", "2021-11-13", "2021-11-14", "NaT"])
    ret = is_chn_busday(x)
    assert np.all(ret == [1, 0, 0, 0, 0])
    ret = not_chn_busday(x)
    assert np.all(ret == [0, 1, 1, 1, 0])


# %%
def test_ChineseHolidaysCalendar():
    holis = np.asarray(list(holidays.keys()), dtype="M8[D]")
    ccal = ChineseHolidaysCalendar()
    assert np.all(holis == ccal.holidays())
