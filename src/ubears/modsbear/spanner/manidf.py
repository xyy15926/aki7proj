#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: manidf.py
#   Author: xyy15926
#   Created: 2024-06-06 11:17:46
#   Updated: 2025-01-21 21:27:34
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
import json
from typing import Any, TypeVar
from collections.abc import Mapping, Callable

import string
import numpy as np
import pandas as pd
from tqdm import tqdm
# from IPython.core.debugger import set_trace

from ubears.flagbear.str2.dups import rename_overlaped

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def group_addup_apply(
    df: pd.DataFrame,
    groupkey: str | list,
    appfunc: Callable,
    desc: str = None,
    **kwargs,
) -> pd.DataFrame:
    """Progress apply for addup-function application.

    Addup functions: Function that won't change the index of the DataFrame but
    just addup some columns in most cases.

    If DF returned by callable applied keep the index unchanged, the duplicates
    in the (index and the group-key) will slow down the process of concatenation.
    So and external `rets` will be used to collect the results of the sub-DF.

    Params:
    -------------------------
    df: DataFrame to be groupe and then be applied `appfunc` on.
    groupkey: Group key.
    appfunc: Application function on groups.
    desc: Description prompt for tqdm progress bar.
    kwargs: Kwargs for `appfunc`.

    Return
    """
    tqdm.pandas(desc=desc)
    rets = []

    # Wrap `appfunc` to collect the return.
    def addup(subdf):
        subret = appfunc(subdf, **kwargs)
        rets.append(subret)
        return 1

    df.groupby(groupkey, sort=False)[df.columns].progress_apply(addup)
    ret = pd.concat(rets)
    return ret


# %%
def merge_dfs(
    dfs: list[pd.DataFrame],
    ons: str | list[list[str]],
    bys: str | list[list[str]] = None,
    hows: str | list[str] = "inner",
    tolerance: int | None = 0,
    direction: str = "nearest",
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
    ons: Field names to join on, which must be found in all DataFrames.
    bys: Field names to match as group before join, which must be found in all
      DataFrames.
      This will only take effect in fuzzy match join situation.
    hows: How to merge.
    tolerance: Tolerance for fuzzy match.
    direction: How to find the match for join key.
      backward:
      forward:
      nearest:

    Return:
    ------------------------
    Merged DataFrame
    """
    # TODO: I forget why to filter empty DataFrame out, but no longer filter.
    # dfs = list(filter(lambda df: not df.empty, dfs))
    if len(dfs) == 1:
        logger.warning("Only 1 no-empty DataFrame passed.")
        return dfs[0]
    elif len(dfs) == 0:
        logger.warning("No no-empty DataFrame passed.")
        return pd.DataFrame()

    ons = [[ons], ] * len(dfs) if np.isscalar(ons) else ons
    if bys is None:
        bys = [None] * len(dfs)
    elif np.isscalar(bys):
        bys = [[bys], ] * len(dfs)
    hows = [hows, ] * (len(dfs) - 1) if np.isscalar(hows) else hows

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
    for rdf, ron, how, rby, rcol_D in zip(dfs[1:], ons[1:], hows, bys[1:], col_Ds[1:]):
        rdf = rdf.sort_values(ron).rename(rcol_D, axis=1)
        if tolerance == 0:
            merged = pd.merge(merged, rdf,
                              left_on=lon, right_on=ron,
                              how=how)
        else:
            merged = pd.merge_asof(merged, rdf,
                                   left_on=lon, right_on=ron,
                                   left_by=lby, right_by=rby,
                                   tolerance=tolerance,
                                   direction=direction)
        lon, lby = ron, rby

    return merged


# %%
def pivot_tags(
    tags: pd.Series,
    sep: str = ",",
) -> pd.DataFrame:
    """Pivot Series with joined tags into DataFrame.

    Split values in `tags` with `seps`, and then count tag frequncies for each
    tag in each record.

    Params:
    -----------------------
    tags: Series with values of tags seperated by `sep`.
    sep: Seperator.

    Return:
    -----------------------
    DataFrame with tags as columns and counts of tags as value.
                tag1    tag2    ...     NULL
        idx1    1       0       ...
        idx2    0       2       ...
        ...
    """
    # Split tags.
    tags = tags.fillna("").astype(str).str.strip(sep).str.split(sep, expand=False)
    tag_counts = (
        pd.DataFrame(
            {
                "id": tags.index.repeat(tags.apply(len)),
                "tags": np.concatenate(tags.values),
                "ones": np.ones(np.add.reduce(tags.apply(len)), dtype=np.int_),
            }
        )
        .replace("", "NULL")
        .groupby(["id", "tags"])["ones"]
        .agg("sum")
        .unstack()
        .fillna(0)
        .astype(np.int_)
    )

    return tag_counts
