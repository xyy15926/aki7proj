#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: calender.py
#   Author: xyy15926
#   Created: 2024-11-09 21:35:31
#   Updated: 2025-02-28 10:40:37
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Any, TypeVar, Tuple
from collections.abc import Callable, Iterator, Sequence
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self

import numpy as np
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from chinese_calendar import holidays


# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
_holidates = list(holidays.keys())
_starty = _holidates[0].year
_endy = _holidates[-1].year
logger.info(f"Holidays from {_starty} to {_endy} are considered.")
if np.datetime64(_holidates[-1], "Y") < np.datetime64("today", "Y"):
    logger.warning("Chinese calender is outdated. "
                   "You may updated the module with pip first.")


# %%
ChnBusdayCalendar = np.busdaycalendar("1111100", list(holidays.keys()))


def is_chn_busday(date: str | Sequence):
    date = np.asarray(date, dtype="M8[D]")
    return np.is_busday(date, busdaycal=ChnBusdayCalendar)


def not_chn_busday(date: str | Sequence):
    date = np.asarray(date, dtype="M8[D]")
    return ~(np.isnat(date) | np.is_busday(date, busdaycal=ChnBusdayCalendar))


# %%
class ChineseHolidaysCalendar(AbstractHolidayCalendar):
    """Chinese Holiday Calendar.

    Read holiday constant from the module `chinese_calendar` for initiation,
    which only record the holiday from 2004 to the current year.
    So remember to update the package annually or it will be outdated.
    """
    rules = [
        Holiday(val, year=key.year,
                month=key.month,
                day=key.day)
        for key, val in holidays.items()]
