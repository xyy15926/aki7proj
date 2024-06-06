#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: manidf.py
#   Author: xyy15926
#   Created: 2024-06-06 11:17:46
#   Updated: 2024-06-06 11:19:08
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
import json
from typing import Any, TypeVar
from collections.abc import Mapping
from collections import ChainMap
import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace

from flagbear.fliper import rename_overlaped

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def merge_dfs(
    dfs: list[pd.DataFrame],
    on: str | list[str],
    by: str | list[str] = None,
    tolerance: int | None = 0,
    direction: str = "nearest",
    *,
    ons: list = None,
    bys: list = None,
) -> pd.DataFrame:
    """Merge DataFrames.

    If tolerance is 0, exact match join will used.
    Else fuzzy match join will be used, which is similar to left-join that the
    nearest key will be matched instead only the exact equal key.
    1. All DataFrame will be sorted by key first.
    2. Join keys must be the exact same dtype, even int the float aren't
      compatiable.

    Params:
    ------------------------
    dfs: DataFrames to be merged.
    on: Field names to join on, which must be found in all DataFrames.
    by: Field names to match as group before join, which must be found in all
      DataFrames.
      This will only take effect in fuzzy match join situation.
    tolerance: Tolerance for fuzzy match.
    direction: How to find the match for join key.
      backward:
      forward:
      nearest:
    ons: Field names to join on for corespondant DataFrame.
    bys: Field names to match as group before join for corespondant DataFrame.

    Return:
    ------------------------
    Merged DataFrame
    """
    assert len(dfs) > 1
    ons = [on, ] * len(dfs) if ons is None else ons
    bys = [by, ] * len(dfs) if bys is None else bys

    # Rename overlaped elements if necessary and then construct the column
    # mapper for each DataFrame with join and group keys excluded.
    ori_colss = [df.columns for df in dfs]
    new_colss = rename_overlaped(ori_colss)
    col_Ds = []
    for ocols, ncols, on_, by_ in zip(ori_colss, new_colss, ons, bys):
        cols_D = {}
        for ocol, ncol in zip(ocols, ncols):
            if (ocol == on_ or (isinstance(on_, list) and ocol in on_)
                    or ocol == by_ or (isinstance(by_, list) and ocol in by_)):
                continue
            cols_D[ocol] = ncol
        col_Ds.append(cols_D)

    lon = ons[0]
    lby = bys[0]
    merged = dfs[0].sort_values(lon).rename(col_Ds[0], axis=1)

    # Merge on by one with `pd.merge_asof` for inexact matching join.
    for rdf, ron, rby, rcol_D in zip(dfs[1:], ons[1:], bys[1:], col_Ds[1:]):
        rdf = rdf.sort_values(ron).rename(rcol_D, axis=1)
        if tolerance == 0:
            merged = pd.merge(merged, rdf,
                              left_on=lon, right_on=ron)
        else:
            merged = pd.merge_asof(merged, rdf,
                                   left_on=lon, right_on=ron,
                                   left_by=lby, right_by=rby,
                                   tolerance=tolerance,
                                   direction=direction)
        lon, lby = ron, rby

    return merged
