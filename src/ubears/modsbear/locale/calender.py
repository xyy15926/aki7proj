#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: calender.py
#   Author: xyy15926
#   Created: 2024-11-09 21:35:31
#   Updated: 2024-11-09 21:36:48
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Any, TypeVar, Tuple
from collections.abc import Callable, Iterator
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self

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
