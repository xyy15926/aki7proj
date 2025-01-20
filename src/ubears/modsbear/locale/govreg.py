#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: govreg.py
#   Author: xyy15926
#   Created: 2024-11-09 21:46:19
#   Updated: 2024-11-21 20:14:18
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Any, TypeVar, Tuple
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self

from functools import lru_cache
import pandas as pd

from ubears.flagbear.slp.finer import get_data

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")

GOVERN_REGION_LV4 = get_data() / "govern_region/govern_region_level4.csv"


# %%
@lru_cache
def get_chn_govrs(deep: int = None):
    """Get get governing region code.

    ATTENTION:
    Not all government code could be cut into 3-level, 469026 for example.
    So usd pid to chain lower and upper level government region may be better?

    Params:
    --------------------------
    deep: Governing region level
      0: Province
      1: City
      2: County
      3: Town

    Return:
    --------------------------
    DataFrame[id, name, PinYin,...]
    """
    if isinstance(deep, int) and (deep < 0 or deep > 3):
        logger.error(f"Unexpected governing level: {deep}.")
        raise ValueError(f"Unexpected governing level: {deep}.")
    reg_df = pd.read_csv(GOVERN_REGION_LV4)
    reg_df["PinYin"] = reg_df["pinyin"].apply(
        lambda x: "".join([ele.capitalize() for ele in x.split(" ")]))

    if deep is None:
        reg_lved = reg_df
    else:
        reg_lved = reg_df[reg_df["deep"] == deep]

    return reg_lved
