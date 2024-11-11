#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: govreg.py
#   Author: xyy15926
#   Created: 2024-11-09 21:46:19
#   Updated: 2024-11-09 21:49:36
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

from flagbear.slp.finer import get_assets_path

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")

GOVERN_REGION_LV4 = get_assets_path() / "govern_region/govern_region_level4.csv"


# %%
@lru_cache
def get_chn_govrs(glv: int = 1):
    """Get get governing region code.

    Params:
    --------------------------
    glv: Governing region level
      1: Province
      2: City
      3: County
      4: Town

    Return:
    --------------------------
    DataFrame[id, name, PinYin,...]
    """
    if glv < 1 or glv > 4:
        logger.error(f"Unexpected governing level: {glv}.")
        raise ValueError(f"Unexpected governing level: {glv}.")
    glvm = {
        0: 0,
        1: 1e2,
        2: 1e4,
        3: 1e6,
        4: 1e8,
        5: 1e11,
    }
    reg_df = pd.read_csv(GOVERN_REGION_LV4)
    reg_df["PinYin"] = reg_df["pinyin"].apply(
        lambda x: "".join([ele.capitalize() for ele in x.split(" ")]))
    reg_lved = reg_df[(reg_df["id"] > glvm[glv - 1])
                      & (reg_df["id"] < glvm[glv])]
    return reg_lved

